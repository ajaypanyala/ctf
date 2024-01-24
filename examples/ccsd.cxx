/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup CCSD CCSD
  * @{ 
  * \brief A Coupled Cluster Singles and Doubles contraction code extracted from Aquarius
  */

#include <ctf.hpp>
using namespace CTF;

// #define SCHEDULE_CCSD 1


void ccsd(int noa, int nva, int nob, int nvb, int ncv, World &dw, int sched_nparts = 0) {
  int rank;   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef SCHEDULE_CCSD
  double timer = MPI_Wtime();
  CTF::Schedule sched(&dw);
  sched.set_max_partitions(sched_nparts);
  
#endif

  //Tensor declarations
  int tshape2[] = {NS,NS};
  int tshape3[] = {NS,NS,NS};
  int tshape4[] = {NS,NS,NS,NS};

  int OaOa[]     = {noa,noa};
  int VaOa[]     = {nva,noa};
  int VaVa[]     = {nva,nva};
  int OaVa[]     = {noa,nva};
  int VbVb[]     = {nvb,nvb};
  int ObOb[]     = {nob,nob};

  int OaVaCv[]   = {noa,nva,ncv};
  int OaOaCv[]   = {noa,noa,ncv};
  int VaOaCv[]   = {nva,noa,ncv};
  int VaVaCv[]   = {nva,nva,ncv};
  int ObObCv[]   = {nob,nob,ncv};
  int VbVbCv[]   = {nvb,nvb,ncv};
  int VbObCv[]   = {nvb,nob,ncv};

  int VaVaOaOa[] = {nva,nva,noa,noa};
  int VaVbOaOb[] = {nva,nvb,noa,nob};
  int OaObOaOb[] = {noa,nob,noa,nob};
  int VaOaVaOa[] = {nva,noa,nva,noa};
  int VbVaObOa[] = {nvb,nva,nob,noa};
  int VbOaVbOa[] = {nvb,noa,nvb,noa};
  int VbOaVaOb[] = {nvb,noa,nva,nob};
  int VbObVbOb[] = {nvb,nob,nvb,nob};
  int VaVbVaVb[] = {nva,nvb,nva,nvb};
  

  Tensor<double> t1_aa(2,VaOa,tshape2,dw); //VaOa
  Tensor<double> f1_oo_aa(2,OaOa,tshape2,dw); //OaOa
  Tensor<double> f1_ov_aa(2,OaVa,tshape2,dw); //OaVa
  Tensor<double> f1_vv_aa(2,VaVa,tshape2,dw); //OaVa
  
  Tensor<double> t2_aaaa(4,VaVaOaOa,tshape4,dw); //VaVaOaOa
  Tensor<double> t2_abab(4,VaVbOaOb,tshape4,dw); //VaVbOaOb
  Tensor<double> t2_aaaa_temp(4,VaVaOaOa,tshape4,dw); //VaVaOaOa

  Tensor<double> i0_aa(2,VaOa,tshape2,dw); //r1_aa
  Tensor<double> i0_abab(4,VaVbOaOb,tshape4,dw); //r2_abab
  Tensor<double> r2_abab(4,VaVbOaOb,tshape4,dw); 

  Tensor<double> chol3d_oo_aa(3,OaOaCv,tshape3,dw); //OaVaCv
  Tensor<double> chol3d_ov_aa(3,OaVaCv,tshape3,dw); //OaVaCv
  Tensor<double> chol3d_vv_aa(3,VaVaCv,tshape3,dw); //VaVaCv

  // Energy tensors
  CTF::Scalar<double> de(0.0,dw);
  CTF::Vector<double> _a01V(ncv,dw); //Cv
  Tensor<double> _a02_aa(3,OaOaCv,tshape3,dw); //OaOaCv
  Tensor<double> _a03_aa(3,OaVaCv,tshape3,dw); //OaVaCv

  // T1 tensors
  CTF::Vector<double> _a02V(ncv,dw); //Cv
  Tensor<double> _a01_aa(3,OaOaCv,tshape3,dw);
  Tensor<double> _a04_aa(2,OaOa,tshape2,dw);
  Tensor<double> _a05_aa(2,OaVa,tshape2,dw);
  Tensor<double> _a06_aa(3,VaOaCv,tshape3,dw);

  // T2 tensors
  CTF::Vector<double> _a007V(ncv,dw); //Cv
  Tensor<double> _a001_aa(2,VaVa,tshape2,dw);
  Tensor<double> _a006_aa(2,OaOa,tshape2,dw);
  Tensor<double> _a008_aa(3,OaOaCv,tshape3,dw);
  Tensor<double> _a009_aa(3,OaOaCv,tshape3,dw);
  Tensor<double> _a017_aa(3,VaOaCv,tshape3,dw);
  Tensor<double> _a021_aa(3,VaVaCv,tshape3,dw);

  Tensor<double> _a001_bb(2,VbVb,tshape2,dw);
  Tensor<double> _a006_bb(2,ObOb,tshape2,dw);
  Tensor<double> _a009_bb(3,ObObCv,tshape3,dw);
  Tensor<double> _a017_bb(3,VbObCv,tshape3,dw);
  Tensor<double> _a021_bb(3,VbVbCv,tshape3,dw);  

  Tensor<double> _a004_aaaa(4,VaVaOaOa,tshape4,dw);
  Tensor<double> _a004_abab(4,VaVbOaOb,tshape4,dw);

  Tensor<double> _a019_abab(4,OaObOaOb,tshape4,dw);
  Tensor<double> _a020_aaaa(4,VaOaVaOa,tshape4,dw);
  Tensor<double> _a020_baba(4,VbOaVbOa,tshape4,dw);
  Tensor<double> _a020_baab(4,VbOaVaOb,tshape4,dw);
  Tensor<double> _a020_bbbb(4,VbObVbOb,tshape4,dw);
  Tensor<double> _a022_abab(4,VaVbVaVb,tshape4,dw);

  Tensor<double> i0_temp(4,VbVaObOa,tshape4,dw);
  

  // sched.record();
  
  /***** CCSD energy terms (closed-shell) ******/
  // t2_aaaa_temp["ijkl"] = 0.0;
  t2_aaaa["ijkl"] = t2_abab["ijkl"]; 
  t2_aaaa_temp["ijkl"] = t2_aaaa["ijkl"]; 
  t2_aaaa["ijkl"] += -1.0 * t2_aaaa_temp["ijkl"];
  t2_aaaa_temp["ijkl"] += t2_aaaa["ijkl"];

  _a01V["c"]      =        t1_aa["ij"] * chol3d_ov_aa["jic"];
  _a02_aa["ijc"]   =        t1_aa["ai"] * chol3d_ov_aa["jac"];
  _a03_aa["iac"]  =        t2_aaaa_temp["abij"] * chol3d_ov_aa["jbc"];
  de[""]          =  2.0 * _a01V["c"] * _a01V["c"];
  de[""]         += -1.0 * _a02_aa["ijc"] * _a02_aa["jic"];
  de[""]         += -1.0 * _a03_aa["iac"] * chol3d_ov_aa["iac"];
  de[""]         +=  2.0 * t1_aa["ai"] * f1_ov_aa["ia"];

  /***** CCSD T1 terms (closed-shell) ******/
  i0_aa["ai"]      =  1.0 * f1_ov_aa["ia"];
  _a01_aa["ijc"]   =  1.0 * t1_aa["aj"] * chol3d_ov_aa["iac"];
  _a02V["c"]       =  2.0 * t1_aa["ai"] * chol3d_ov_aa["iac"];
  _a05_aa["ia"]    = -1.0 * chol3d_ov_aa["jac"] * _a01_aa["ijc"];
  _a05_aa["ia"]   +=  1.0 * f1_ov_aa["ia"];

  _a06_aa["aic"]   = -1.0 * t2_aaaa_temp["abij"] * chol3d_ov_aa["jbc"];
  _a04_aa["ji"]    =  1.0 * f1_oo_aa["ji"];
  _a04_aa["ji"]   +=  1.0 * chol3d_ov_aa["jac"] * _a06_aa["aic"];
  _a04_aa["ji"]   += -1.0 * t1_aa["ai"] * f1_ov_aa["ja"];
  i0_aa["ai"]     +=  1.0 * t1_aa["aj"] * _a04_aa["ji"];
  i0_aa["aj"]     +=  1.0 * chol3d_ov_aa["jac"] * _a02V["c"];
  i0_aa["aj"]     +=  1.0 * t2_aaaa_temp["abji"] * _a05_aa["ib"];
  i0_aa["bi"]     += -1.0 * chol3d_vv_aa["bac"] * _a06_aa["aic"];
  _a06_aa["bjc"]  += -1.0 * t1_aa["aj"] * chol3d_vv_aa["bac"];
  i0_aa["aj"]     += -1.0 * _a06_aa["ajc"] * _a02V["c"];
  _a06_aa["bic"]  += -1.0 * t1_aa["bi"] * _a02V["c"];
  _a06_aa["bic"]  +=  1.0 * t1_aa["bj"] * _a01_aa["jic"];
  _a01_aa["jic"]  +=  1.0 * chol3d_oo_aa["jic"];
  i0_aa["bi"]     +=  1.0 * _a01_aa["jic"] * _a06_aa["bjc"];
  i0_aa["bi"]     +=  1.0 * t1_aa["ai"] * f1_vv_aa["ba"];

  /***** CCSD T2 terms (closed-shell) ******/
  _a017_aa["ajc"]     =  -1.0 * t2_aaaa_temp["abji"] * chol3d_ov_aa["ibc"];
  _a006_aa["ji"]      =  -1.0 * chol3d_ov_aa["jbc"] * _a017_aa["bic"];
  _a007V["c"]         =   2.0 * chol3d_ov_aa["iac"] * t1_aa["ai"];
  _a009_aa["ijc"]     =   1.0 * chol3d_ov_aa["iac"] * t1_aa["aj"];
  _a021_aa["bac"]     =  -0.5 * chol3d_ov_aa["iac"] * t1_aa["bi"];
  _a021_aa["bac"]    +=   0.5 * chol3d_vv_aa["bac"];
  _a017_aa["ajc"]    +=  -2.0 * t1_aa["bj"] * _a021_aa["abc"];
  _a008_aa["ijc"]     =   1.0 * _a009_aa["ijc"];
  _a009_aa["jic"]    +=   1.0 * chol3d_oo_aa["ijc"];
  _a009_bb["jic"]     =  _a009_aa["jic"];
  _a021_bb["bac"]     =  _a021_aa["bac"];
  _a001_aa["ab"]      =  -2.0 * _a021_aa["abc"] * _a007V["c"];
  _a001_aa["ab"]     +=  -1.0 * _a017_aa["ajc"] * chol3d_ov_aa["jbc"];
  _a006_aa["ji"]     +=   1.0 * _a009_aa["jic"] * _a007V["c"];
  _a006_aa["ki"]     +=  -1.0 * _a009_aa["jic"] * _a008_aa["kjc"];

  _a019_abab["jmin"]  =  0.25 * _a009_aa["jic"] * _a009_bb["mnc"];
  _a020_aaaa["bjai"]  = -2.0  * _a009_aa["jic"] * _a021_aa["bac"];
  _a020_baba["djci"]  =         _a020_aaaa["djci"];
  _a020_aaaa["akej"] +=  0.5  * _a004_aaaa["beki"] * t2_aaaa["abij"];
  _a020_baab["cjan"]  = -0.5  * _a004_aaaa["baji"] * t2_abab["bcin"];
  _a020_baba["cidj"] +=  0.5  * _a004_abab["adim"] * t2_abab["acjm"];
  _a017_aa["ajc"]    +=  1.0  * t1_aa["ai"] * chol3d_oo_aa["ijc"];
  _a017_aa["ajc"]    += -1.0  * chol3d_ov_aa["jac"];
  _a001_aa["ba"]     += -1.0  * f1_vv_aa["ba"];
  _a001_aa["ba"]     +=  1.0  * t1_aa["bi"] * f1_ov_aa["ia"];
  _a006_aa["ji"]     +=  1.0  * f1_oo_aa["ji"];
  _a006_aa["ji"]     +=  1.0  * t1_aa["ai"] * f1_ov_aa["ja"];

  _a017_bb["cmv"]     = _a017_aa["cmv"];
  _a006_bb["mn"]      = _a006_aa["mn"];
  _a001_bb["cd"]      = _a001_aa["cd"];
  _a021_bb["cdv"]     = _a021_aa["cdv"];
  _a020_bbbb["cmdn"]  = _a020_aaaa["cmdn"];

  i0_abab["adjm"]     =  1.0 * _a020_bbbb["dncm"] * t2_abab["acjn"];
  i0_abab["bcjm"]    +=  1.0 * _a020_baab["ciam"] * t2_aaaa["baji"];
  i0_abab["acjm"]    +=  1.0 * _a020_baba["cidj"] * t2_abab["adim"];
  i0_temp["cani"]     =  i0_abab["cani"];
  i0_abab["acjm"]    +=  1.0 * i0_temp["camj"];
  i0_abab["acin"]    +=  1.0 * _a017_aa["aiv"] * _a017_bb["cnv"];
  _a022_abab["adbc"]  =  1.0 * _a021_aa["abv"] * _a021_bb["dcv"];
  i0_abab["adin"]    +=  4.0 * _a022_abab["adbc"] * t2_abab["bcin"];
  _a019_abab["jmin"] +=  0.25 * _a004_abab["adjm"] * t2_abab["adin"];
  i0_abab["acin"]    +=  4.0 * _a019_abab["jmin"] * t2_abab["acjm"];
  i0_abab["acin"]    += -1.0 * t2_abab["adin"] * _a001_bb["cd"];
  i0_abab["acin"]    += -1.0 * t2_abab["bcin"] * _a001_aa["ab"];
  i0_abab["acjm"]    += -1.0 * t2_abab["acim"] * _a006_aa["ij"];
  i0_abab["acin"]    += -1.0 * t2_abab["acim"] * _a006_bb["mn"];

#ifdef SCHEDULE_CCSD 
  if (rank == 0) {
    printf("Record: %lf\n",
            MPI_Wtime()-timer);
  }

  timer = MPI_Wtime();
  CTF::ScheduleTimer schedule_time = sched.execute();
#endif


#ifdef SCHEDULE_CCSD
  if (rank == 0) {
    printf("Schedule comm down: %lf\n", schedule_time.comm_down_time);
    printf("Schedule execute: %lf\n", schedule_time.exec_time);
    printf("Schedule imbalance, wall: %lf\n", schedule_time.imbalance_wall_time);
    printf("Schedule imbalance, accum: %lf\n", schedule_time.imbalance_acuum_time);
    printf("Schedule comm up: %lf\n", schedule_time.comm_up_time);
    printf("Schedule total: %lf\n", schedule_time.total_time);
    printf("All execute: %lf\n",
            MPI_Wtime()-timer);
  }
#endif
} 

#ifndef TEST_SUITE

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, niter, noa, nva, nob, nvb, ncv, sched_nparts, i;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-noa")){
    noa = atoi(getCmdOption(input_str, input_str+in_num, "-noa"));
    if (noa < 0) noa = 10;
  } else noa = 10;
  if (getCmdOption(input_str, input_str+in_num, "-nva")){
    nva = atoi(getCmdOption(input_str, input_str+in_num, "-nva"));
    if (nva < 0) nva = 6;
  } else nva = 6;
  if (getCmdOption(input_str, input_str+in_num, "-ncv")){
    ncv = atoi(getCmdOption(input_str, input_str+in_num, "-ncv"));
    if (ncv < 0) ncv = 10;
  } else ncv = 10;  
  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 1;
  } else niter = 1;
  if (getCmdOption(input_str, input_str+in_num, "-nparts")){
    sched_nparts = atoi(getCmdOption(input_str, input_str+in_num, "-nparts"));
    if (sched_nparts < 0) sched_nparts = 0;
  } else sched_nparts = 0;

  // TODO: open-shell
  nob = noa;
  nvb = nva;

  if (rank == 0)
    printf("noa=%d, nva=%d, nob=%d, nvb=%d, ncv=%d, niter=%d\n", noa,nva,nob,nvb,ncv,niter);

  {
    World dw(argc, argv);
    {
      Timer_epoch tccsd("CCSD");
      tccsd.begin();
      for (i=0; i<niter; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        double d = MPI_Wtime();
        ccsd(noa, nva, nob, nvb, ncv, dw, sched_nparts);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
          printf("(%d nodes) Completed %dth CCSD iteration in time = %lf\n",
              np, i, MPI_Wtime()-d);
      }
      tccsd.end();
    }
  }

  MPI_Finalize();
  return 0;
}

#endif

