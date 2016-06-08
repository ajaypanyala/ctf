#include "csr.h"
#include "../contraction/ctr_comm.h"
#include "../shared/util.h"

#define ALIGN 256

namespace CTF_int {
  int64_t get_csr_size(int64_t nnz, int nrow, int val_size){
    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz*val_size;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nrow+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += sizeof(int)*nnz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return offset;
  }

  CSR_Matrix::CSR_Matrix(int64_t nnz, int nrow, int ncol, algstrct const * sr){
    ASSERT(ALIGN >= 16);
    int64_t size = get_csr_size(nnz, nrow, sr->el_size);
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = nnz;
    ((int64_t*)all_data)[1] = sr->el_size;
    ((int64_t*)all_data)[2] = nrow;
    ((int64_t*)all_data)[3] = ncol;
  }

  CSR_Matrix::CSR_Matrix(char * all_data_){
    ASSERT(ALIGN >= 16);
    all_data = all_data_;
  }

  CSR_Matrix::CSR_Matrix(COO_Matrix const & coom, int nrow, int ncol, algstrct const * sr, char * data){
    ASSERT(ALIGN >= 16);
    int64_t nz = coom.nnz(); 
    int64_t v_sz = coom.val_size(); 
    int const * coo_rs = coom.rows();
    int const * coo_cs = coom.cols();
    char const * vs = coom.vals();

    int64_t size = get_csr_size(nz, nrow, v_sz);
    if (data == NULL)
      all_data = (char*)alloc(size);
    else
      all_data = data;
    ((int64_t*)all_data)[0] = nz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = nrow;
    ((int64_t*)all_data)[3] = ncol;

    char * csr_vs = vals();
    int * csr_rs = rows();
    int * csr_cs = cols();

    //memcpy(csr_vs, vs, nz*v_sz);
    //memset(csr_rs

    sr->coo_to_csr(nz, nrow, csr_vs, csr_cs, csr_rs, vs, coo_rs, coo_cs);
/*    for (int i=0; i<nrow; i++){
      printf("csr_rs[%d] = %d\n",i,csr_rs[i]);
    }
    for (int i=0; i<nz; i++){
      printf("csr_cs[%d] = %d\n",i,csr_cs[i]);
    }*/
    
  }

  int64_t CSR_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int CSR_Matrix::val_size() const {
    return ((int64_t*)all_data)[1];
  }


  int64_t CSR_Matrix::size() const {
    return get_csr_size(nnz(),nrow(),val_size());
  }
  
  int CSR_Matrix::nrow() const {
    return ((int64_t*)all_data)[2];
  }
  
  int CSR_Matrix::ncol() const {
    return ((int64_t*)all_data)[3];
  }
  
  char * CSR_Matrix::vals() const {
    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return all_data + offset;
  }

  int * CSR_Matrix::rows() const {
    int64_t n = this->nnz();
    int v_sz = this->val_size();

    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);

    return (int*)(all_data + offset);
  } 

  int * CSR_Matrix::cols() const {
    int64_t n = this->nnz();
    int64_t nr = this->nrow();
    int v_sz = this->val_size();

    int offset = 4*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nr+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    //return (int*)(all_data + n*v_sz+(nr+1)*sizeof(int)+3*sizeof(int64_t));
    return (int*)(all_data + offset);
  } 

  void CSR_Matrix::csrmm(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_gemm && do_offload){
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
      func->coffload_csrmm(m,n,k,A,B,C);
    } else {
      CSR_Matrix cA((char*)A);
      int64_t nz = cA.nnz(); 
      int const * rs = cA.rows();
      int const * cs = cA.cols();
      char const * vs = cA.vals();
      if (func != NULL && func->has_gemm){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->ccsrmm(m,n,k,vs,cs,rs,nz,B,C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_A->csrmm(m,n,k,alpha,vs,cs,rs,nz,B,beta,C,func);
      }
    }
  }

  void CSR_Matrix::csrmultd(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    assert(func == NULL);
  }

  CSR_Matrix * CSR_Matrix::partition(int s, char ** parts_buffer){
    int part_nnz[s], part_nrows[s];
    int m = nrow();
    char * org_vals = vals();
    int * org_rows = rows();
    int * org_cols = cols();
    for (int i=0; i<s; i++){
      part_nnz[i] = 0;
      part_nrows[i] = 0;
    }
    for (int i=0; i<m; i++){
      part_nrows[i%s]++;
      part_nnz[i]+=org_rows[i+1]-org_rows[i];
    }
    int64_t tot_sz = 0;
    for (int i=0; i<s; i++){
      tot_sz += get_csr_size(part_nnz[i], part_nrows[i], val_size());
    }
    char * new_data = (char*)alloc(tot_sz);
    char * part_data = new_data;
    CSR_Matrix * parts = (CSR_Matrix*)alloc(sizeof(CSR_Matrix)*s);
    for (int i=0; i<s; i++){
      ((int64_t*)part_data)[0] = part_nnz[i];
      ((int64_t*)part_data)[1] = val_size();
      ((int64_t*)part_data)[2] = part_nrows[i];
      ((int64_t*)part_data)[3] = ncol();
      parts[i] = CSR_Matrix(part_data);
      char * pvals = parts[i].vals();
      int * prows = parts[i].rows();
      int * pcols = parts[i].cols();
      prows[0] = 1;
      for (int j=i; j<m; j+=s){
        memcpy(pvals+(prows[j/s]-1)*val_size(), org_vals+(org_rows[j]-1)*val_size(), (org_rows[j+1]-org_rows[j])*val_size());
        memcpy(pcols+(prows[j/s]-1)*sizeof(int), org_cols+(org_rows[j]-1)*sizeof(int), (org_rows[j+1]-org_rows[j])*sizeof(int));
        prows[j/s+1] = prows[j/s]+org_rows[j+1]-org_rows[j];
      }
      part_data += get_csr_size(part_nnz[i], part_nrows[i], val_size());
    }
    return parts;
  }
      
  CSR_Matrix::CSR_Matrix(char * const * smnds, int s){
    CSR_Matrix csrs[s];
    int64_t tot_nnz=0, tot_nrow=0;
    for (int i=0; i<s; i++){
      csrs[i] = CSR_Matrix(smnds[i]);
      tot_nnz += csrs[i].nnz();
      tot_nrow += csrs[i].nrow();
    }
    int64_t v_sz = csrs[0].val_size();
    int64_t tot_ncol = csrs[0].ncol();
    all_data = (char*)alloc(get_csr_size(tot_nnz, tot_nrow, v_sz));
    ((int64_t*)all_data)[0] = tot_nnz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = tot_nrow;
    ((int64_t*)all_data)[3] = tot_ncol;
    
    char * csr_vs = vals();
    int * csr_rs = rows();
    int * csr_cs = cols();

    csr_rs[0] = 1;

    for (int i=0; i<tot_nrow; i++){
      int ipart = i%s;
      int const * prows = csrs[ipart].rows();
      int i_nnz = prows[i/s+1]-prows[i/s];
      memcpy(csr_vs+(csr_rs[i]-1)*v_sz,
             csrs[ipart].vals()+(prows[i/s]-1)*v_sz,
             i_nnz*v_sz);
      memcpy(csr_cs+(csr_rs[i]-1)*sizeof(int),
             csrs[ipart].cols()+(prows[i/s]-1)*sizeof(int),
             i_nnz*sizeof(int));
      csr_rs[i+1] = csr_rs[i]+i_nnz;
    }
  }

}
