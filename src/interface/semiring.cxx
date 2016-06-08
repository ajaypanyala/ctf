#include "set.h"
#include "../shared/blas_symbs.h"
#include "../shared/offload.h"


namespace CTF_int {
  template <>
  void default_axpy<float>
                   (int           n,
                    float         alpha,
                    float const * X,
                    int           incX,
                    float *       Y,
                    int           incY){
    CTF_BLAS::SAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy<double>
                   (int            n,
                    double         alpha,
                    double const * X,
                    int            incX,
                    double *       Y,
                    int            incY){
    CTF_BLAS::DAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<float> >
                   (int                         n,
                    std::complex<float>         alpha,
                    std::complex<float> const * X,
                    int                         incX,
                    std::complex<float> *       Y,
                    int                         incY){
    CTF_BLAS::CAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<double> >
                   (int                          n,
                    std::complex<double>         alpha,
                    std::complex<double> const * X,
                    int                          incX,
                    std::complex<double> *       Y,
                    int                          incY){
    CTF_BLAS::ZAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_scal<float>(int n, float alpha, float * X, int incX){
    CTF_BLAS::SSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal<double>(int n, double alpha, double * X, int incX){
    CTF_BLAS::DSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<float> >
      (int n, std::complex<float> alpha, std::complex<float> * X, int incX){
    CTF_BLAS::CSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<double> >
      (int n, std::complex<double> alpha, std::complex<double> * X, int incX){
    CTF_BLAS::ZSCAL(&n,&alpha,X,&incX);
  }

#define DEF_COOMM_KERNEL()                                \
    for (int j=0; j<n; j++){                              \
      for (int i=0; i<m; i++){                            \
        C[j*m+i] *= beta;                                 \
      }                                                   \
    }                                                     \
    for (int i=0; i<nnz_A; i++){                          \
      int row_A = rows_A[i]-1;                            \
      int col_A = cols_A[i]-1;                            \
      for (int col_C=0; col_C<n; col_C++){                \
         C[col_C*m+row_A] += alpha*A[i]*B[col_C*k+col_A]; \
      }                                                   \
    }

  template <>
  void default_coomm< float >
          (int           m,
           int           n,
           int           k,
           float         alpha,
           float const * A,
           int const *   rows_A,
           int const *   cols_A,
           int           nnz_A,
           float const * B,
           float         beta,
           float *       C){
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_SCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (float*)A, rows_A, cols_A, &nnz_A,
               (float*)B, &k, &beta,
               (float*)C, &m);
#else
    DEF_COOMM_KERNEL();
#endif
  }

  template <>
  void default_coomm< double >
          (int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           int const *    rows_A,
           int const *    cols_A,
           int            nnz_A,
           double const * B,
           double         beta,
           double *       C){
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    TAU_FSTART(MKL_DCOOMM);
    CTF_BLAS::MKL_DCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (double*)A, rows_A, cols_A, &nnz_A,
               (double*)B, &k, &beta,
               (double*)C, &m);
    TAU_FSTOP(MKL_DCOOMM);
#else
    DEF_COOMM_KERNEL();
#endif
  }


  template <>
  void default_coomm< std::complex<float> >
          (int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           int const *                 rows_A,
           int const *                 cols_A,
           int                         nnz_A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C){
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_CCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (std::complex<float>*)A, rows_A, cols_A, &nnz_A,
               (std::complex<float>*)B, &k, &beta,
               (std::complex<float>*)C, &m);
#else
    DEF_COOMM_KERNEL();
#endif
  }

  template <>
  void default_coomm< std::complex<double> >
     (int                          m,
      int                          n,
      int                          k,
      std::complex<double>         alpha,
      std::complex<double> const * A,
      int const *                  rows_A,
      int const *                  cols_A,
      int                          nnz_A,
      std::complex<double> const * B,
      std::complex<double>         beta,
      std::complex<double> *       C){
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_ZCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (std::complex<double>*)A, rows_A, cols_A, &nnz_A,
               (std::complex<double>*)B, &k, &beta,
               (std::complex<double>*)C, &m);
#else
    DEF_COOMM_KERNEL();
#endif
  }
/*
#if USE_SP_MKL
  template <>
  bool get_def_has_csrmm<float>(){ return true; }
  template <>
  bool get_def_has_csrmm<double>(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<float> >(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<double> >(){ return true; }
#else
  template <>
  bool get_def_has_csrmm<float>(){ return true; }
  template <>
  bool get_def_has_csrmm<double>(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<float> >(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<double> >(){ return true; }
#endif
*/

#if USE_SP_MKL
  template <>  
  void def_coo_to_csr<float>(int64_t nz, int nrow, float * csr_vs, int * csr_ja, int * csr_ia, float const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    CTF_BLAS::MKL_SCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (float*)coo_vs, coo_rs, coo_cs, &info);

  }
  template <>  
  void def_coo_to_csr<double>(int64_t nz, int nrow, double * csr_vs, int * csr_ja, int * csr_ia, double const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    TAU_FSTART(MKL_DCSRCOO);
    CTF_BLAS::MKL_DCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (double*)coo_vs, coo_rs, coo_cs, &info);
    TAU_FSTOP(MKL_DCSRCOO);
  }
  template <>  
  void def_coo_to_csr<std::complex<float>>(int64_t nz, int nrow, std::complex<float> * csr_vs, int * csr_ja, int * csr_ia, std::complex<float> const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    CTF_BLAS::MKL_CCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (std::complex<float>*)coo_vs, coo_rs, coo_cs, &info);
  }
  template <>  
  void def_coo_to_csr<std::complex<double>>(int64_t nz, int nrow, std::complex<double> * csr_vs, int * csr_ja, int * csr_ia, std::complex<double> const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    CTF_BLAS::MKL_ZCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (std::complex<double>*)coo_vs, coo_rs, coo_cs, &info);
  }

  bool try_mkl_coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_ja, int * csr_ia, char const * coo_vs, int const * coo_rs, int const * coo_cs, int el_size){
    switch (el_size){
      case 4:
        def_coo_to_csr<float>(nz,nrow,(float*)csr_vs,csr_ja,csr_ia,(float const*)coo_vs,coo_rs,coo_cs);
        break;
      case 8:
        def_coo_to_csr<double>(nz,nrow,(double*)csr_vs,csr_ja,csr_ia,(double const*)coo_vs,coo_rs,coo_cs);
        break;
      case 16:
        def_coo_to_csr<std::complex<double>>(nz,nrow,(std::complex<double>*)csr_vs,csr_ja,csr_ia,(std::complex<double> const*)coo_vs,coo_rs,coo_cs);
        break;
    } 
    return true; 
  }
#else
  template <> 
  void def_coo_to_csr<float>(int64_t nz, int nrow, float * csr_vs, int * csr_ja, int * csr_ia, float const * coo_vs, int const * coo_rs, int const * coo_cs){
    seq_coo_to_csr<float>(nz, nrow, csr_vs, csr_ja, csr_ia, coo_vs, coo_rs, coo_cs);
  }
  template <>  
  void def_coo_to_csr<double>(int64_t nz, int nrow, double * csr_vs, int * csr_ja, int * csr_ia, double const * coo_vs, int const * coo_rs, int const * coo_cs){
    seq_coo_to_csr<double>(nz, nrow, csr_vs, csr_ja, csr_ia, coo_vs, coo_rs, coo_cs);
  }
  template <>  
  void def_coo_to_csr<std::complex<float>>(int64_t nz, int nrow, std::complex<float> * csr_vs, int * csr_ja, int * csr_ia, std::complex<float> const * coo_vs, int const * coo_rs, int const * coo_cs){
    seq_coo_to_csr<std::complex<float>>(nz, nrow, csr_vs, csr_ja, csr_ia, coo_vs, coo_rs, coo_cs);
  }
  template <>  
  void def_coo_to_csr<std::complex<double>>(int64_t nz, int nrow, std::complex<double> * csr_vs, int * csr_ja, int * csr_ia, std::complex<double> const * coo_vs, int const * coo_rs, int const * coo_cs){
    seq_coo_to_csr<std::complex<double>>(nz, nrow, csr_vs, csr_ja, csr_ia, coo_vs, coo_rs, coo_cs);
  }

  bool try_mkl_coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_ja, int * csr_ia, char const * coo_vs, int const * coo_rs, int const * coo_cs, int el_size){
    return 0; 
  }
#endif
#if (USE_SP_MKL!=1)
  template <typename dtype>
  void muladd_csrmm
                 (int           m,
                  int           n,
                  int           k,
                  dtype         alpha,
                  dtype const * A,
                  int const *   JA,
                  int const *   IA,
                  int           nnz_A,
                  dtype const * B,
                  dtype         beta,
                  dtype *       C){
    TAU_FSTART(muladd_csrmm);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int row_A=0; row_A<m; row_A++){
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int col_B=0; col_B<n; col_B++){
        C[col_B*m+row_A] *= beta;
        if (IA[row_A] < IA[row_A+1]){
          int i_A1 = IA[row_A]-1;
          int col_A1 = JA[i_A1]-1;
          dtype tmp = A[i_A1]*B[col_B*k+col_A1];
          for (int i_A=IA[row_A]; i_A<IA[row_A+1]-1; i_A++){
            int col_A = JA[i_A]-1;
            tmp += A[i_A]*B[col_B*k+col_A];
          }
          C[col_B*m+row_A] += alpha*tmp;
        }
      }
    }
    TAU_FSTOP(muladd_csrmm);
  }

  template<typename dtype>
  void muladd_csrmultd
                 (int           m,
                  int           n,
                  int           k,
                  dtype const * A,
                  int const *   JA,
                  int const *   IA,
                  int           nnz_A,
                  dtype const * B,
                  int const *   IB,
                  int const *   JB,
                  int           nnz_B,
                  dtype *       C){
    TAU_FSTART(muladd_csrmultd);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int row_A=0; row_A<m; row_A++){
      for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
        int row_B = JA[i_A]-1; //=col_A
        for (int i_B=IB[row_B]-1; i_B<IB[row_B+1]-1; i_B++){
          int col_B = JB[i_B]-1;
          C[col_B*m+row_A] += A[i_A]*B[j_B];
        }
      }
    }
    TAU_FSTOP(muladd_csrmultd);
  }
#endif
}

namespace CTF {

  template <>
  void CTF::Semiring<float,1>::default_csrmm
          (int           m,
           int           n,
           int           k,
           float         alpha,
           float const * A,
           int const *   JA,
           int const *   IA,
           int           nnz_A,
           float const * B,
           float         beta,
           float *       C) const {
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_SCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm<float>(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }

  template <>
  void CTF::Semiring<double,1>::default_csrmm
          (int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           int const *    JA,
           int const *    IA,
           int            nnz_A,
           double const * B,
           double         beta,
           double *       C) const {
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    TAU_FSTART(MKL_DCSRMM);
    CTF_BLAS::MKL_DCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
    TAU_FSTOP(MKL_DCSRMM);
#else
    CTF_int::muladd_csrmm<double>(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }


  template <>
  void CTF::Semiring<std::complex<float>,0>::default_csrmm
          (int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           int const *                 JA,
           int const *                 IA,
           int                         nnz_A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C) const {
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_CCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm< std::complex<float> >(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }

  template <>
  void CTF::Semiring<std::complex<double>,0>::default_csrmm
          (int                          m,
           int                          n,
           int                          k,
           std::complex<double>         alpha,
           std::complex<double> const * A,
           int const *                  JA,
           int const *                  IA,
           int                          nnz_A,
           std::complex<double> const * B,
           std::complex<double>         beta,
           std::complex<double> *       C) const {
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_ZCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm< std::complex<double> >(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }

  template<>
  void CTF::Semiring<double,1>::default_csrmultd
                 (int            m,
                  int            n,
                  int            k,
                  double const * A,
                  int const *    JA,
                  int const *    IA,
                  int            nnz_A,
                  double const * B,
                  int const *    JB,
                  int const *    IB,
                  int            nnz_B,
                  double *       C) const {
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    TAU_FSTART(MKL_DCSRMULTD);
    CTF_BLAS::MKL_DCSRMULTD(&transa, &m, &n, &k, A, JA, IA, B, JB, IB, C, &m);
    TAU_FSTOP(MKL_DCSRMULTD);
#else
    CTF_int::muladd_csrmultd<double>(m,n,k,A,JA,IA,nnz_A,B,JB,IB,nnz_B,C);
#endif

  }

/*  template<> 
  bool CTF::Semiring<float,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<float>;
  }*/
  template<> 
  bool CTF::Semiring<float,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<float>;
  }

  template<> 
  bool CTF::Semiring<std::complex<float>,0>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm< std::complex<float> >;
  }


  template<> 
  bool CTF::Semiring<double,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<double>;
  }

  template<> 
  bool CTF::Semiring<std::complex<double>,0>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm< std::complex<double> >;
  }

  template<> 
  void CTF::Semiring<double,1>::offload_gemm(
                        char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
    int lda_A = k;
    if (tA == 'n' || tA == 'N') lda_A = m;
    int lda_B = n;
    if (tB == 'N' || tB == 'N') lda_B = k;
    CTF_int::offload_gemm<double>(tA, tB, m, n, k, ((double const*)alpha)[0], (double const *)A, lda_A, (double const *)B, lda_B, ((double const*)beta)[0], (double*)C, m);
  }

  template<> 
  void CTF::Semiring<std::complex<double>,0>::offload_gemm(
                        char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
    int lda_A = k;
    if (tA == 'n' || tA == 'N') lda_A = m;
    int lda_B = n;
    if (tB == 'N' || tB == 'N') lda_B = k;
    CTF_int::offload_gemm<std::complex<double>>(tA, tB, m, n, k, ((std::complex<double> const*)alpha)[0], (std::complex<double> const *)A, lda_A, (std::complex<double> const *)B, lda_B, ((std::complex<double> const*)beta)[0], (std::complex<double>*)C, m);
  }


}
