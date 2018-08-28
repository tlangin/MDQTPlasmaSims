
/***********************************************************************/
/*                                                                     */
/*  MD simulation of ions interacting via screened Coulomb forces      */
/*  initial conditions correspond to zero temperature and flat density */
/*  but can be changed to accomodate finite velocities                 */
/*  and initial density perturbations                                  */
/*                                                                     */
/*  We have also added the ability to simulate the laser cooling of    */
/*  the Sr+ ion, including all 12 sublevels and S->P and D->P lasers   */
/*  Along with the ability to do the cooling in an expanding frame     */
/*  by adding a time dependent detuning                                */
/*                                                                     */
/*                                                                     */
/***********************************************************************/


/******************************************************************************************************************************************/
/*                                                                                                                                        */
/*    To Compile: first at some point you must load the g++ compiler and c++11 by typing "module load GCC/4.9.3" Then, to compile, type   */
/*                                                                                                                                        */
/*    g++ -std=c++11 -fopenmp -o runFile -O3 LaserCoolWithExpansion.cpp -lm -I/home/tkl1/usr/include -L/users/tkl1/user/lib64             */
/*                                                                                                                                        */
/*    where you should replace "tkl1" with whatever your user name, and you can choose whatever name you want for "runFile"               */
/*    which is the "executable" file created from the compilation process                                                                 */
/*                                                                                                                                        */
/******************************************************************************************************************************************/




#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<omp.h>
#include<sys/stat.h>
#include<iostream>
#include<complex>
#include<armadillo>
#include<random>

using namespace std;
using namespace arma;
std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> uni(0, 1);
auto random_double = uni(rng);
std::complex<double> I(0,1);
/* GLOBAL VARIABLES */

/* save Directory */

char saveDirectory[256] = "dataLaserCoolTestDoppShift/";//main directory.  A subfolder titled  Gamma_Kappa_Number___ will be created.  Subsubfolders for each job will be created.  data will be stored in the job folders;

/*input variables: the only ones you'll ever really want to change*/


const int N0 =1000;                // average particle number in simulation cell
double detuning=-0.5;//SP detuning normalized by \gamma_{SP}
double Om=0.5;//SP Rabi Freq norm by \gamma_{SP}
double tmax = 45000;
bool applyForce=true;

/* other input variables: You'll probably never want to change these*/
int nrthread = 4;              // number of threads
int sampleFreq = 1000;		   // output data for all functions every X timesteps
#define TIMESTEP 0.01		   // default time step

unsigned job;                  // job number, just use as label
double dt;                     // time step
double t;                      // actual time
int c0;
/* output variables */
unsigned counter=0;            // time counter, used as output-file label
double PvelX[2001];          // velocity distribution
double PvelY[2001];          // velocity distribution
double PvelZ[2001];          // velocity distribution
double vel[2001];           // corresponding velocity bins

/* system variables */
const double temperature = 0.01; //(Temp in K)
std::normal_distribution<double> velocityDistribution(0, 1.0508*sqrt(temperature));//set up velocity generator, spread given by sqrt(T_norm)...i.e. sqrt(1/Gamma)
double V[3][N0+1000];        // ion velocities

/*QUANTUM STUFF*/
//cooling parameters
double vKick = 0.0012076;
//waveFunctions
cx_mat wvFns[N0+1000];
double tPart[N0+1000];
double numStates = 3;
mat ident = mat(numStates,numStates,fill::eye);
cx_mat wvFn1=cx_mat(ident.col(0),mat(numStates,1,fill::zeros));//"Ground State" J=0
cx_mat wvFn2=cx_mat(ident.col(1),mat(numStates,1,fill::zeros));//"Excited state" mj=+1
cx_mat wvFn3=cx_mat(ident.col(2),mat(numStates,1,fill::zeros));//other "excited state" mj=-1


//decay coupling and rates
cx_mat cs[2];
double gs[2];


/* FUNCTIONS */ // you don't actually have to predeclare these, so there may be some "missing"
void init(void);                     // system initialization
void qstep(void);                    // advance quantum part of system
void output(void);                   // output results

/***********************************************************************/
/*                                                                     */
/*  system initialization                                              */
/*                                                                     */
/***********************************************************************/

void init(void)
{
    int i;

    for(i=0;i<N0;i++)              // loop over particles in large box
      {
	
	V[0][i] = velocityDistribution(rng);
	V[1][i] = velocityDistribution(rng);
	V[2][i] = velocityDistribution(rng);
	wvFns[i]= cx_mat(ident.col(0),0*ident.col(0));
	tPart[i]=0.;
        
      }


}


/***********************************************************************/
/*                                                                     */
/*  advance quantum system                                               */
/*                                                                     */
/***********************************************************************/

void qstep(void)
{  
  unsigned i,j;
  cx_mat wvFn;
  double velQuant;
  double velPlas;
  //double tPart;
  mat zero_mat1=mat(1,1,fill::zeros);
  cx_mat dpmat;
  double kick;
  cx_mat densMatrix;
  //density matrix terms for force calculation
  cx_mat p13;
  cx_mat p12;
  //hamiltonian and various terms
  double totalDetRight,totalDetLeft,dp,rand,dtHalf,prefactor;
  cx_mat hamCouplingTerm,hamEnergyTerm,hamDecayTerm,hamWithoutDecay,hamil;
  cx_mat matPrefactor,wvFnStepped,k1,k2,k3,k4,wvFnk1,wvFnk2,wvFnk3;
  cx_mat dpmatTerms[2];
  double rand2,rand3,norm3,norm4,norm5,norm6,totalNorm,prob3,prob4,prob5,prob6,randDir;

  /*begin parallel*/  //yeah i know it's a lot of variables...

#pragma omp parallel private(i,j,wvFn,velQuant,velPlas,kick,densMatrix,p12,p13,totalDetRight,totalDetLeft,dp,rand,dtHalf,prefactor,hamCouplingTerm,hamEnergyTerm,hamDecayTerm,dpmat,hamWithoutDecay,hamil,matPrefactor,wvFnStepped,k1,k2,k3,k4,wvFnk1,wvFnk2,wvFnk3,dpmatTerms,randDir) shared(V,wvFns,cs,wvFn1,wvFn2,wvFn3,vKick,Om,detuning,gs,I,numStates,zero_mat1,tPart)
  {
#pragma omp for 
  
    for(i=0;i<N0;i++){//for every wavefunction: evolve it
    dpmat=cx_mat(zero_mat1,zero_mat1);
    wvFn=wvFns[i];
    velQuant=V[0][i];//use x velocity
    tPart[i] += dt;
    for(j=0;j<2;j++){
      dpmatTerms[j] = dt*wvFn.t()*cs[j].t()*cs[j]*wvFn;
      //dpmatTerms[j].print("dpTerms:");
      //dpmat.print("dpMat:");
      //zero_mat1.print("zeroMat:");
      dpmat=dpmat+dpmatTerms[j]*gs[j];
      
    }//for all states, calculate dpmat
    dp = dpmat(0,0).real();
    rand = drand48();
    if(rand>dp)//if no jump, evolve according to non-Hermitian Hamiltonian (see Lukin book or TKL PhD Thesis)
      {
	//first calculate the force
	densMatrix = wvFn*wvFn.t();
	p13 = wvFn1.t()*densMatrix*wvFn3;//coupling between 2 (S: mJ=+1/2) and 3 (P: mJ=3/2), etc.
	p12 = wvFn1.t()*densMatrix*wvFn2;
	
	kick = 1*vKick*Om*(p13(0,0).imag()*sqrt(gs[0])-p12(0,0).imag()*sqrt(gs[1]))*dt;//dhange in velocity due to quantum force, see TKL thesis for full calculation
	
	//next evolve the wavefunction: first calculate the hamiltonian and various terms
	totalDetRight = -detuning-velQuant;//propegating leftward, from right
	totalDetLeft = -detuning+velQuant;//expDetuning same sign as vel detuning...after all it comes from a velocity
	hamCouplingTerm = -Om/2*wvFn1*wvFn3.t()*sqrt(gs[0])-Om/2*wvFn1*wvFn2.t()*sqrt(gs[1]);
        
	hamEnergyTerm = totalDetRight*(wvFn3*wvFn3.t())+totalDetLeft*(wvFn2*wvFn2.t());//energy terms are of form "Energy" X |n\rangle \langle 
      
    
	hamWithoutDecay=hamEnergyTerm+hamCouplingTerm+hamCouplingTerm.t();//add all the non-decay terms together, including hermitian conjugates of coupling terms!
	
	hamDecayTerm=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the decay type terms
	for(j=0;j<2;j++){
	  hamDecayTerm=hamDecayTerm-1./2*I*(gs[j]*cs[j].t()*cs[j]);
	}
	
	//cx_mat hamDecayTerm(zeros<mat>(numStates,numStates),-1*hamDecayTermComplex);
	
	hamil=hamWithoutDecay+hamDecayTerm;//total Hamiltonian for non-hermitian evolution
	
	//with hamiltonian calculated, can evolve wvFn using RK method (I choose 3/8 method)

	dtHalf = dt/2;
	
	//get k1,y1 (k1 is slope at t0 calculated using y0, y0 is initial wvFn value.  y1 (wvFnk1) is wvFn stepped by dt/2 w/ slope k1)
	dpmatTerms[2];
	dpmat=cx_mat(zero_mat1,zero_mat1);
	for(j=0;j<2;j++){
	  dpmatTerms[j] = dt*wvFn.t()*cs[j].t()*cs[j]*wvFn;
	  dpmat=dpmat+dpmatTerms[j]*gs[j];
	}
        dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
	matPrefactor = ident-I*dt*hamil;
	wvFnStepped = prefactor*matPrefactor*wvFn;
	k1 = 1./(dt)*(wvFnStepped-wvFn);
	wvFnk1 = wvFn+dtHalf*k1;
	
	//get k2,y2 (k2 is slope at t0+dt/2 calculated using y1, y2 (wvFnk2) is wvFn stepped by dt/2 w/ slope k2)
	dpmatTerms[2];
	dpmat=cx_mat(zero_mat1,zero_mat1);
	for(j=0;j<2;j++){
	  dpmatTerms[j] = dt*wvFnk1.t()*cs[j].t()*cs[j]*wvFnk1;
	  dpmat=dpmat+dpmatTerms[j]*gs[j];
	}
	dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
	matPrefactor = ident-I*dt*hamil;
	wvFnStepped = prefactor*matPrefactor*wvFnk1;
	k2 = 1./(dt)*(wvFnStepped-wvFnk1);
	wvFnk2 = wvFn+dtHalf*k2;

	
	//get k3, y3 (k3 is slope at t0+dt/2 calculated using y2, y3 (wvFnk3) is wvFn stepped by dt w/ slope k3)
	
	dpmatTerms[2];
        dpmat=cx_mat(zero_mat1,zero_mat1);
	for(j=0;j<2;j++){
	  dpmatTerms[j] = dt*wvFnk2.t()*cs[j].t()*cs[j]*wvFnk2;
	  dpmat=dpmat+dpmatTerms[j]*gs[j];
	}
        dp = dpmat(0,0).real();
        prefactor = 1/sqrt(1-dp);
	matPrefactor = ident-I*dt*hamil;
	wvFnStepped = prefactor*matPrefactor*wvFnk2;
	k3 = 1./(dt)*(wvFnStepped-wvFnk2);
	wvFnk3 = wvFn+dt*k3;
	
	//get k4, yfinal (k4 is slope at t0+dt calculated using y3, yfinal is wvFn stepped by dt using weighted average of k1,k2,k3, and k4)
	
	dpmatTerms[2];
	dpmat=cx_mat(zero_mat1,zero_mat1);
	for(j=0;j<2;j++){
	  dpmatTerms[j] = dt*wvFnk3.t()*cs[j].t()*cs[j]*wvFnk3;
	  dpmat=dpmat+dpmatTerms[j]*gs[j];
	}
	dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
	matPrefactor = ident-I*dt*hamil;
	wvFnStepped = prefactor*matPrefactor*wvFnk3;
	k4 = 1./(dt)*(wvFnStepped-wvFnk3);
	wvFn = wvFn+(k1+3*k2+3*k3+k4)/8*(dt);//finally: evolve the wavefunction according to completion of runge-kutta propagator
      }
    else{//else if there was a "jump" roll again for which state was "jumped" into
      tPart[i]=0;
      wvFn.zeros();
      wvFn(0,0).real(1);
	
      randDir = drand48();
      if(randDir<0.5){//give the kick corresponding to a 408 photon
	kick=vKick;
      }
      else{
	kick=-vKick;
      }
    }
      
    if(applyForce){
      V[0][i]=V[0][i]+kick;
    }
    wvFns[i] = wvFn;
    }//for all particles, evolve wave function
  }//pragmaOmp
}


void output(void)
{
  unsigned i,j;
  double V2;         // width of gaussian weight function for Pvel(vel)
    double velXAvg=0.0;
    double EkinX,EkinY,EkinZ;       // average kinetic energy along each direction
    FILE *fa;
    FILE *fa2;
    FILE *fa3;
    char dataDirCopy[256];
    char buffer[256];
    char fileName[256];
    
    EkinX=0.0;

    for(i=0;i<N0;i++)   // calculate total kinetic energy (separate x from y,z)
    {
      EkinX+=0.5*((V[0][i])*(V[0][i]));

    }
    EkinX/=(double)N0;   // normalize by ion number
    
    // output energies (2. & 3. column) and energy change (4. column) and avg Vel;
    strcpy(dataDirCopy,saveDirectory);
    strcpy(fileName,strcat(dataDirCopy,"energies.dat"));
    fa = fopen(fileName,"a");
    fprintf(fa,"%lg\t%lg\n",t,EkinX);
    fclose(fa);

    /*
    //output ion states vs vel (column1 vx, column2 total S state, column 3 total P state, etc.)
    double currVel,popS,popP,popD;
    cx_mat currWvFn;
    strcpy(dataDirCopy,saveDirectory);
    snprintf(buffer, 256,"statePopulationsVsVTime%06d.dat", c0);
    strcpy(fileName,strcat(dataDirCopy,buffer));
    fa=fopen(fileName, "w");
    for (i=0;i<N0;i++){
      currVel=V[0][i];
      currWvFn = wvFns[i];
      popS = std::norm(currWvFn(0,0)) +std::norm(currWvFn(1,0));
      popP = std::norm(currWvFn(2,0)) + std::norm(currWvFn(3,0)) + std::norm(currWvFn(4,0)) + std::norm(currWvFn(5,0));
      popD = std::norm(currWvFn(6,0)) + std::norm(currWvFn(7,0)) + std::norm(currWvFn(8,0)) + std::norm(currWvFn(9,0)) + std::norm(currWvFn(10,0)) + std::norm(currWvFn(11,0));
      fprintf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\n",currVel,popS,popP,popD,popS+popP+popD);
    }
    fclose(fa);
 
    */
    /**********************************/
    /* ADD MORE OBSERVABLES TO OUTPUT */
    /**********************************/
}


/***********************************************************************/
/*                                                                     */
/*  main routine                                                       */
/*                                                                     */
/***********************************************************************/

int main(int argc, char **argv)
{


  //setup directory structure
  
  job=(unsigned)atof(argv[1]);       // input job label
  //make new main directory
  mkdir(saveDirectory,ACCESSPERMS);
  //make new sub directory
  char namebuf0[256];
  char namebuf[256];
  char namebuf2[256];
  char saveDirBackup[256];
  //strcpy(saveDirBackup,saveDirectory);
  sprintf(namebuf0,"Om%d/",(unsigned)(Om*100));
  strcat(saveDirectory,namebuf0);
  mkdir(saveDirectory,ACCESSPERMS);
  sprintf(namebuf,"Det%dNumIons%dInitialTemp%duK",(unsigned)(detuning*100),(unsigned)N0,(unsigned)(temperature*1000000));
  strcat(saveDirectory,namebuf);
  mkdir(saveDirectory,ACCESSPERMS);
  //make directory for given job
  sprintf(namebuf2,"/job%d/",job);
  strcat(saveDirectory,namebuf2);
  mkdir(saveDirectory,ACCESSPERMS);
  //saveDirectory is now of form "OriginalSaveDirectory/Gamma%d...etc/job1/"
  
  //have to define all this here for some reason instead of the global varaible section...
  cs[0] = wvFn1*wvFn2.t();
  cs[1] = wvFn1*wvFn3.t();
  gs[0]=1;
  gs[1]=1;

  srand48((unsigned)time(NULL)+job); // initialize random number generator
  
  init();
  t=0;
  dt=0.01;
  c0=0;
  while(t<=tmax)                     // run simulation until tmax
    {
      t+=dt;
      // other outputs like energy, vel_dist
      if ((c0+1)%sampleFreq == 0)
	{
	  output();
	}

      
      qstep();
      c0++;
    }
 return 0;
}
