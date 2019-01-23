
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
/*    To Compile on davinci cluster: first at some point you must load the g++ compiler and c++11  and armadillo by entering:             */
/*                                                                                                                                        */
/*    1) module load GCC/5.4.0                                                                                                            */
/*    2) module load OpenMPI/1.10.3                                                                                                       */
/*    3) module load Armadillo/7.600.1                                                                                                    */
/*                                                                                                                                        */
/*    Then, to compile, type                                                                                                              */
/*                                                                                                                                        */
/*    g++ -std=c++11 -fopenmp -o runFile -O3 LaserCoolWithExpansion.cpp -lm -larmadillo                                                   */
/*                                                                                                                                        */
/*    where you can choose whatever name you want for "runFile"                                                                           */
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

/* GLOBAL VARIABLES */

/* save Directory */

char saveDirectory[256] = "dataLaserCool/";//main directory.  A subfolder titled  Gamma_Kappa_Number___ will be created.  Subsubfolders for each job will be created.  data will be stored in the job folders;

/*input variables: the only ones you'll ever really want to change*/

double Ge = 0.1;			   // kappa = sqrt(3*Ge)
int newRun = 1;			       // Is it a new run (1), or is it starting from annealed conditions (0)?
int c0 = 0;				   // timestep counter, if starting from annealed conditions, this should be the same as the timestep label on the input files (see files labeled "conditions")
#define tmax 30                // maximum simulation time: on 01/22/19 tested for 3000 particles: a sim for tmax = 45 concluded in 7 hrs, so to keep your sims under 8 hours simulate no more than t= 50/(N/3000)^2 where N is particle number at a time...obviously this depends on number of particles and on density (lower density means more quantum steps per MD step means LONGER Time)
//(note: time IS loaded for annealed conditions...so for example if you do a new sim with tmax=2 and want to continue from the ending conditions up to t=4, you'll need to set tmax =4)
double density = 2;              //units of 10^14 m^-3
double sig0=4.0;                   //initial width in units mm
double Te=19.0;                    //in units kelvin
double fracOfSig=0;            //location of plasma chunk in fractions of sigma.  make this 0 for a non moving frame
#define N0 3500                // average particle number in simulation cell
double detuning=-1;//SP detuning normalized by \gamma_{SP}
double detuningDP=1;//DP detuning normalized by \gamma_{SP} (not a typo: all "quantum times" normalized by \gamma_{SP}, see TKL PhD Thesis
double Om=1;//SP Rabi Freq norm by \gamma_{SP}
double OmDP=1;//DP Rabi Frequency nrom by \gamma_{SP}
bool reNormalizewvFns=false;

/* other input variables: You'll probably never want to change these*/
int nrthread = 4;              // number of threads
int sampleFreq = 40;		   // output data for all functions every X timesteps
double gamToEinsteinFreq = 174.07/sqrt(density);// ratio of gamma=1.41e8 to einstein freq
#define TIMESTEP 0.002		   // default time step
#define lambdaFRAC 12           // 1+max value of a single integer in the k integer triplet
#define tstartC0 0.88          // MUST BE GREATER THAN 0.02
int plasmaToQuantumTimestepRatio = (int) ceil(34.81/sqrt(density));//1/5 of above quantity because MD timestep is 0.002omega-1 while quantum timestep of 0.01gamma-1
double quantumTimestep = TIMESTEP/plasmaToQuantumTimestepRatio;//timestep for quantum evolution (0.01Gam) in plasma time units
double plasVelToQuantVel=1.1821*pow(density,1./6);//conversion factor for going from plasma to quantum velocities (norm by a\omega and k/\gamma respectively)


//starting invervals for calculating VAF
 // need to change main() if you want to add intervals							  
#define tstartV0 3			  
#define tstartV1 5
#define tstartV2 7
#define tstartV3 9
#define tstartV4 11
#define tstartV5 13
#define tstartV6 15
#define tstartV7 17
#define tstartV8 19
#define tstartV9 21
#define tstartV10 23
#define tstartV11 25
#define tstartV12 27

#define numberOfIntervalC 1
#define numberOfIntervalV 13
#define lengthOfIntervalC 100000 
// number of time steps to output J
#define lengthOfIntervalV 100000  // number of time steps for V intervals

double lDeb;                   // electron CCP and Debye length
double L;                      // size of simulation cell
unsigned job;                  // job number, just use as label
double dt;                     // time step
double t;                      // actual time

/* output variables */
unsigned counter=0;            // time counter, used as output-file label
double Epot;               // current potential energy
double Epot0;              // initial potential energy
double PvelX[2001];          // velocity distribution
double PvelY[2001];          // velocity distribution
double PvelZ[2001];          // velocity distribution
double vel[2001];           // corresponding velocity bins

/* system variables */
double R[3][N0+1000];        // ion positions
double V[3][N0+1000];        // ion velocities
double F[3][N0+1000];        // forces for each ion
unsigned N;             // number of particles

/* Velocity Autocorrelation Function */
double VAF;     								     // velocity autocorrelation function
double Vholder[3][numberOfIntervalV][N0+1000];       // Hold Velocity data

/* Longitudinal Current Correlation Function */
std::complex<double> J[3][lambdaFRAC][lambdaFRAC][lambdaFRAC]; // Fourier transform of current operator
std::complex<double> I(0,1);

/* Average velocity of spin up subset */
double velocityBin[300];
double velocityProb[300];
int number[numberOfIntervalV];

/*QUANTUM STUFF*/
//cooling parameters
double decayRatioD5Halves=0.0617;
double kRat = 0.395;
double vKick = 0.001208/plasVelToQuantVel;
double vKickDP = vKick*kRat;
//waveFunctions
cx_mat wvFns[N0+1000];
double tPart[N0+1000];
double numStates = 12;
mat ident = mat(numStates,numStates,fill::eye);
//naming convention same as in TKL thesis:  wvFn1=|1\rangle = S mJ=-1/2, etc.
cx_mat wvFn1=cx_mat(ident.col(0),mat(numStates,1,fill::zeros));//S mJ=-1/2
cx_mat wvFn2=cx_mat(ident.col(1),mat(numStates,1,fill::zeros));//S mJ=+1/2
cx_mat wvFn3=cx_mat(ident.col(2),mat(numStates,1,fill::zeros));//P mJ=+3/2
cx_mat wvFn4=cx_mat(ident.col(3),mat(numStates,1,fill::zeros));//P mJ=+1/2
cx_mat wvFn5=cx_mat(ident.col(4),mat(numStates,1,fill::zeros));//P mJ=-1/2
cx_mat wvFn6=cx_mat(ident.col(5),mat(numStates,1,fill::zeros));//P mJ=-3/2
cx_mat wvFn7=cx_mat(ident.col(6),mat(numStates,1,fill::zeros));//D mJ=-5/2
cx_mat wvFn8=cx_mat(ident.col(7),mat(numStates,1,fill::zeros));//D mJ=-3/2
cx_mat wvFn9=cx_mat(ident.col(8),mat(numStates,1,fill::zeros));//D mJ=-1/2
cx_mat wvFn10=cx_mat(ident.col(9),mat(numStates,1,fill::zeros));//D mJ=+1/2
cx_mat wvFn11=cx_mat(ident.col(10),mat(numStates,1,fill::zeros));//D mJ=+3/2
cx_mat wvFn12=cx_mat(ident.col(11),mat(numStates,1,fill::zeros));//D mJ=+5/2 (NOTE: only take decay to D_{5/2} into account
//I added the 3/2 D state and repumping via 422 at one point, but it didn't really affect anything and made the sim take way longer...)

//decay coupling and rates
cx_mat cs[18];
double gs[18];
cx_mat hamDecayTerm=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the decay type terms
cx_mat decayMatrix = cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the coupling type terms
cx_mat hamCouplingTermNoTimeDep=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the coupling type terms

/* FUNCTIONS */ // you don't actually have to predeclare these, so there may be some "missing"
void init(void);                     // system initialization
void forces(void);                   // calculate all forces
void step_V(double DT);              // advance velocities
void step_R(double DT);              // advance positions
void step(void);                     // advance whole system
void qstep(void);                    // advance quantum part of system
void Epotential(void);               // calculate potential energy
void output(void);                   // output results
/***********************************************************************/
/*                                                                     */
/*  calculate all forces                                               */
/*                                                                     */
/***********************************************************************/

void forces(void)
{
    double rx,ry,rz,dx,dy,dz,dr,fx,fy,fz,ftotal; // all necessary variables for computation
	double Rcut = L/2.;

    int i,j;
	
	for(i = 0; i<N; i++)
	{
		F[0][i] = 0.; F[1][i] = 0.; F[2][i] = 0.;
	}
/* begin parallel */
    #pragma omp parallel private(i,j,rx,ry,rz,dx,dy,dz,dr,fx,fy,fz,ftotal) shared(F,R,N,L,Rcut,lDeb)
	{
    #pragma omp for
    	for(i=0;i<N-1;i++) // loop over all ions i
    	{
    		fx = fy = fz = 0.;
    		rx = R[0][i]; ry = R[1][i]; rz = R[2][i];
        	for(j=i+1;j<N;j++) // loop over all other ions j
        	{
				dx=rx-R[0][j]; // distance between ion i and j
   				dy=ry-R[1][j];
   				dz=rz-R[2][j];

   				// minimum-image criterion
   				dx -= L*round(dx/L);
 				dy -= L*round(dy/L);
   				dz -= L*round(dz/L);
   				dr = sqrt(dx*dx + dy*dy + dz*dz);
				if (dr > 0 && dr < Rcut)
				{
					ftotal = (1./dr + 1./lDeb)*exp(-dr/lDeb)/(dr*dr);    //Yukawa force
					fx=dx*ftotal;
					fy=dy*ftotal;
					fz=dz*ftotal;
					F[0][i] += fx; F[0][j] -= fx;
					F[1][i] += fy; F[1][j] -= fy;
					F[2][i] += fz; F[2][j] -= fz;
   				}
  	     	}
		}
    }
/* end parallel */
}

/***********************************************************************/
/*                                                                     */
/*  calculate potential energy                                         */
/*                                                                     */
/***********************************************************************/

void Epotential(void)
{
	double rx,ry,rz,dx,dy,dz,dr; // all necessary variables for computation
	double Rcut = L/2.;

    int i,j;
	Epot = 0.;
    	for(i=0;i<N;i++)
    	{   
    		rx = R[0][i]; ry = R[1][i]; rz = R[2][i];
        	for(j=i+1;j<N;j++)
        	{
				dx=rx-R[0][j]; // distance between ion i and j
   				dy=ry-R[1][j];
   				dz=rz-R[2][j];

   				// minimum-image criterion
   				dx -= L*round(dx/L);
 				dy -= L*round(dy/L);
   				dz -= L*round(dz/L);
   				
   				dr = sqrt(dx*dx + dy*dy + dz*dz);
				if (dr > 0 && dr < Rcut)
				{
					Epot += exp(-dr/lDeb)/(dr);
   				}
  	     	}
		}
    Epot/=(double)N; // normalize to obtain potential energy per particle
    
    /***********************************************************************/
    /*                                                                     */
    /*  This neglects the self-energy,                                     */
    /*  which should be small for a sufficient box size.                   */
    /*  So, this serves as good test for proper parameter choice           */
    /*                                                                     */
    /***********************************************************************/
}

/***********************************************************************/
/*                                                                     */
/*  system initialization                                              */
/*                                                                     */
/***********************************************************************/

void init(void)
{
    int i;
    double x,y,z;
    double N9L;
    
    lDeb=1./sqrt(3.*Ge);                           // Debye length in units of a
    
    L=pow(N0*4.*M_PI/3.,0.333333333);              // box length for N0 ions
    
    N9L=(unsigned)(9.*9.*9.*(L*L*L)*3./(4.*M_PI)); // particle number in large box of length 9*L
    
    N=0;                                           // initialize actual particle number
    
    for(i=0;i<N9L;i++)              // loop over particles in large box
    {
        x=9.*L*drand48()-4.*L;      // sample random positions centered around simulation cell
        y=9.*L*drand48()-4.*L;
        z=9.*L*drand48()-4.*L;
        if(x<=L && y<=L && z<=L && x>0 && y>0 && z>0)
        {                           // if particle falls into simulation cell:
            R[0][N]=x;              // store its position
            R[1][N]=y;
            R[2][N]=z;
            V[0][N]=0.;             // store its velocity
            V[1][N]=0.;
            V[2][N]=0.;

	    double rand1 = drand48();
	    double rand2 = drand48();
	    double rand3=drand48();
	    double sign=1;
	    if(rand3<0.5){
	      sign =-1;
	    }
	    double rand4=drand48();
	    double sign2=1;
	    if(rand4<0.5){
	      sign2=-1;
	    }
	    mat wvFn1Cont = sqrt(rand1)*ident.col(0);
	    mat wvFn2RealCont = sign2*sqrt(1-rand1)*sqrt(rand2)*ident.col(1);
	    mat wvFn2ImCont = sign*sqrt(1-rand1)*sqrt(1-rand2)*ident.col(1);
	    wvFns[N]=cx_mat(wvFn1Cont+wvFn2RealCont,wvFn2ImCont);
	    tPart[N]=0;

            N++;                    // increase particle number by 1
        }
    }
    printf("%i\n",N);
    
    for(i=0;i<2001;i++)           // set up bins for velocity distribution
    {                            // bin size is chosen as 0.0025 and range [-5:5]
        vel[i]=(double)i*0.0025;

    }
	Epotential();
	Epot0 = Epot;
    c0 = -1;
}

/***********************************************************************/
/*                                                                     */
/*  advance positions                                                  */
/*                                                                     */
/***********************************************************************/

void step_R(double DT)
{
    unsigned i;
    
    if(t>0)                      // if system has past initial time step
    {
        for(i=0;i<N;i++)         // use leap-frog
        {
            R[0][i]+=DT*V[0][i];
            R[1][i]+=DT*V[1][i];
            R[2][i]+=DT*V[2][i];
        }
    }
    
    else                         // if not
    {
      //forces();//forces calculated in MAIN
        for(i=0;i<N;i++)         // use 2. order integrator
        {
            R[0][i]+=DT*V[0][i]+DT*DT*F[0][i];
            R[1][i]+=DT*V[1][i]+DT*DT*F[1][i];
            R[2][i]+=DT*V[2][i]+DT*DT*F[2][i];
        }
    }
    
    for(i=0;i<N;i++)             // reinsert ions that left the box
    {
      if(R[0][i]<0){R[0][i]+=L;}
      if(R[0][i]>L){R[0][i]-=L;}
      if(R[1][i]<0){R[1][i]+=L;}
      if(R[1][i]>L){R[1][i]-=L;}
      if(R[2][i]<0){R[2][i]+=L;}
      if(R[2][i]>L){R[2][i]-=L;}
    }
}

/***********************************************************************/
/*                                                                     */
/*  advance velocities                                                 */
/*                                                                     */
/***********************************************************************/

void step_V(double DT)
{
    unsigned i;
    
    //forces(); (NOW forces is handled with a call in main!)
    for(i=0;i<N;i++)
    {
        V[0][i]+=DT*F[0][i];  // use leap-frog
        V[1][i]+=DT*F[1][i];
        V[2][i]+=DT*F[2][i];
    }
}

/***********************************************************************/
/*                                                                     */
/*  advance plasma system                                               */
/*                                                                     */
/***********************************************************************/


void step(void)
{
  //if(t<0.02){dt=0.002;}     // at early times use smaller timestep, because of larger forces
  //else{dt=TIMESTEP;}            // later use larger timestep
  //dt=TIMESTEP;
  //dt=quantumTimestep*plasmaToQuantumTimestepRatio;
  dt = quantumTimestep;//step is now called for every QUANTUM timestep
  step_R(0.5*dt);       // leap-frog, symplectic integrator
  //t+=0.5*dt;//quantum system is shorter timescale: qstep now handles time evolution.
  step_V(dt);
  step_R(0.5*dt);
  //t+=0.5*dt;//quantum system is shorter timescale: qstep now handles time evolution.
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
  mat zero_mat1=mat(1,1,fill::zeros);
  cx_mat dpmat;
  double dtQuant = quantumTimestep;
  double expDetuning = 0.0126*fracOfSig*Te*t/(sqrt(density)*sig0*sqrt(1+0.00014314*t*t*Te/(density*sig0*sig0)));
  double kick;
  cx_mat densMatrix;
  //density matrix terms for force calculation
  cx_mat p23;
  cx_mat p14;
  cx_mat p25;
  cx_mat p16;
  cx_mat p96;
  cx_mat p105;
  cx_mat p114;
  cx_mat p123;
  cx_mat p76;
  cx_mat p85;
  cx_mat p94;
  cx_mat p103;
  //hamiltonian and various terms
  double totalDetRightSP,totalDetLeftSP,dp,rand,dtHalf,prefactor;
  cx_mat hamCouplingTerm,hamEnergyTermP,hamEnergyTermD,hamEnergyTerm,hamWithoutDecay,hamil;
  cx_mat matPrefactor,wvFnStepped,k1,k2,k3,k4,wvFnk1,wvFnk2,wvFnk3;
  cx_mat dpmatTerms[18];
  double rand2,rand3,norm3,norm4,norm5,norm6,totalNorm,prob3,prob4,prob5,prob6,randDOrS,randDir;
  bool sDecay;
  double popS,popP,popD;

  /*begin parallel*/  //yeah i know it's a lot of variables...

#pragma omp parallel private(i,j,wvFn,velQuant,velPlas,kick,densMatrix,p23,p14,p25,p16,p96,p105,p114,p123,p76,p85,p94,p103,totalDetRightSP,totalDetLeftSP,dp,rand,dtHalf,prefactor,hamCouplingTerm,hamEnergyTermP,hamEnergyTermD,hamEnergyTerm,dpmat,hamWithoutDecay,hamil,matPrefactor,wvFnStepped,k1,k2,k3,k4,wvFnk1,wvFnk2,wvFnk3,dpmatTerms,rand2,rand3,norm3,norm4,norm5,norm6,totalNorm,prob3,prob4,prob5,prob6,randDOrS,randDir,sDecay,popS,popP,popD) shared(V,R,N,wvFns,plasVelToQuantVel,gamToEinsteinFreq,cs,wvFn1,wvFn2,wvFn3,wvFn4,wvFn5,wvFn6,wvFn7,wvFn8,wvFn9,wvFn10,wvFn11,wvFn12,vKick,Om,detuning,detuningDP,OmDP,decayRatioD5Halves,gs,I,kRat,numStates,zero_mat1,dtQuant,expDetuning,tPart,hamCouplingTermNoTimeDep,decayMatrix,hamDecayTerm)
  {
#pragma omp for 
  
    for(i=0;i<N;i++){//for every wavefunction: evolve it
      dpmat=cx_mat(zero_mat1,zero_mat1);
      wvFn=wvFns[i];
      velPlas=V[0][i];//use x velocity, lasers going along x dir, velocity in PLASMA units
      velQuant=velPlas*plasVelToQuantVel;//convert velocity to quantum units
      tPart[i]+=dtQuant;
      dpmat = dtQuant*gamToEinsteinFreq*wvFn.t()*decayMatrix*wvFn;
      dp = dpmat(0,0).real();
      rand = drand48();
    if(rand>dp)//if no jump, evolve according to non-Hermitian Hamiltonian (see Lukin book or TKL PhD Thesis)
      {
	//first calculate the force
	densMatrix = wvFn*wvFn.t();
	p23 = wvFn2.t()*densMatrix*wvFn3;//coupling between 2 (S: mJ=+1/2) and 3 (P: mJ=3/2), etc.
	p14 = wvFn1.t()*densMatrix*wvFn4;
	p25 = wvFn2.t()*densMatrix*wvFn5;
	p16 = wvFn1.t()*densMatrix*wvFn6;
	p96 = wvFn9.t()*densMatrix*wvFn6;
	p105 = wvFn10.t()*densMatrix*wvFn5;
	p114 = wvFn11.t()*densMatrix*wvFn4;
	p123 = wvFn12.t()*densMatrix*wvFn3;
	p76 = wvFn7.t()*densMatrix*wvFn6;
	p85 = wvFn8.t()*densMatrix*wvFn5;
	p94 = wvFn9.t()*densMatrix*wvFn4;
	p103 = wvFn10.t()*densMatrix*wvFn3;
	kick = 1*vKick*Om*(p23(0,0).imag()*gs[0]+p14(0,0).imag()*gs[2]-p25(0,0).imag()*gs[4]-p16(0,0).imag()*gs[5])*dtQuant*gamToEinsteinFreq+vKickDP*(OmDP/decayRatioD5Halves)*(p96(0,0).imag()*gs[8]+p105(0,0).imag()*gs[11]+p114(0,0).imag()*gs[14]+p123(0,0).imag()*gs[17]-p76(0,0).imag()*gs[6]-p85(0,0).imag()*gs[9]-p94(0,0).imag()*gs[12]-p103(0,0).imag()*gs[15])*dtQuant*gamToEinsteinFreq;//dhange in velocity due to quantum force, see TKL thesis for full calculation
	
	//next evolve the wavefunction: first calculate the hamiltonian and various terms
	totalDetRightSP = -detuning-velQuant-expDetuning;//propegating leftward, from right
	totalDetLeftSP = -detuning+velQuant+expDetuning;//expDetuning same sign as vel detuning...after all it comes from a velocity
	hamCouplingTerm = hamCouplingTermNoTimeDep - OmDP/2*wvFn9*wvFn6.t()*gs[8]/sqrt(decayRatioD5Halves)*exp(I*2.*(velQuant+expDetuning)*(1+kRat)*tPart[i]*gamToEinsteinFreq) - OmDP/2*wvFn10*wvFn5.t()*gs[11]/sqrt(decayRatioD5Halves)*exp(I*2.*(expDetuning+velQuant)*(1+kRat)*tPart[i]*gamToEinsteinFreq);
	hamEnergyTermP = totalDetRightSP*(wvFn3*wvFn3.t()+wvFn4*wvFn4.t())+totalDetLeftSP*(wvFn5*wvFn5.t()+wvFn6*wvFn6.t());//energy terms are of form "Energy" X |n\rangle \langle 
	hamEnergyTermD = (-detuning+detuningDP+(1-kRat)*(velQuant+expDetuning))*(wvFn7*wvFn7.t()+wvFn8*wvFn8.t())+(-detuning+detuningDP+(kRat-1)*(velQuant+expDetuning))*(wvFn11*wvFn11.t()+wvFn12*wvFn12.t())+(-detuning+detuningDP-velQuant-expDetuning-kRat*(velQuant+expDetuning))*(wvFn9*wvFn9.t()+wvFn10*wvFn10.t());
	hamEnergyTerm=hamEnergyTermP+hamEnergyTermD;
        if(i==0){
	  //(hamCouplingTerm+hamCouplingTerm.t()).print("HamCoup");
	  //hamDecayTerm.print("HamDec:");
	}
    
	hamWithoutDecay=hamEnergyTerm+hamCouplingTerm+hamCouplingTerm.t();//add all the non-decay terms together, including hermitian conjugates of coupling terms!
	
	//cx_mat hamDecayTerm(zeros<mat>(numStates,numStates),-1*hamDecayTermComplex);
	
	hamil=hamWithoutDecay+hamDecayTerm;//total Hamiltonian for non-hermitian evolution
	
	//with hamiltonian calculated, can evolve wvFn using RK method (I choose 3/8 method)

	dtHalf = dtQuant*gamToEinsteinFreq/2;
	matPrefactor = ident-I*dtQuant*gamToEinsteinFreq*hamil;
	dpmat=cx_mat(zero_mat1,zero_mat1);
	
	//get k1,y1 (k1 is slope at t0 calculated using y0, y0 is initial wvFn value.  y1 (wvFnk1) is wvFn stepped by dt/2 w/ slope k1)
	dpmat = dtQuant*gamToEinsteinFreq*wvFn.t()*decayMatrix*wvFn;
	dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
      
	wvFnStepped = prefactor*matPrefactor*wvFn;
	k1 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFn);
	wvFnk1 = wvFn+dtHalf*k1;
	
	
	
	//get k2,y2 (k2 is slope at t0+dt/2 calculated using y1, y2 (wvFnk2) is wvFn stepped by dt/2 w/ slope k2)
	dpmat = dtQuant*gamToEinsteinFreq*wvFnk1.t()*decayMatrix*wvFnk1;
	dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
	wvFnStepped = prefactor*matPrefactor*wvFnk1;
	k2 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFnk1);
	wvFnk2 = wvFn+dtHalf*k2;
      
	
	
	//get k3, y3 (k3 is slope at t0+dt/2 calculated using y2, y3 (wvFnk3) is wvFn stepped by dt w/ slope k3)

	dpmat = dtQuant*gamToEinsteinFreq*wvFnk2.t()*decayMatrix*wvFnk2;
	dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
	wvFnStepped = prefactor*matPrefactor*wvFnk2;
	k3 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFnk2);
	wvFnk3 = wvFn+dtQuant*gamToEinsteinFreq*k3;
	
	
	//get k4, yfinal (k4 is slope at t0+dt calculated using y3, yfinal is wvFn stepped by dt using weighted average of k1,k2,k3, and k4)

	dpmat = dtQuant*gamToEinsteinFreq*wvFnk3.t()*decayMatrix*wvFnk3;
	dp = dpmat(0,0).real();
	prefactor = 1/sqrt(1-dp);
	wvFnStepped = prefactor*matPrefactor*wvFnk3;
	k4 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFnk3);
	wvFn = wvFn+(k1+3*k2+3*k3+k4)/8*(dtQuant*gamToEinsteinFreq);//finally: evolve the wavefunction according to completion of runge-kutta propagator
	
	
      }

    
    else{//else if there was a "jump" roll again for which state was "jumped" into
      tPart[i]=0;
      rand2 = drand48();
      norm3 = std::norm(wvFn(2,0));
      norm4 = std::norm(wvFn(3,0));
      norm5 = std::norm(wvFn(4,0));
      norm6 = std::norm(wvFn(5,0));
      totalNorm = norm3+norm4+norm5+norm6;
      prob3=norm3/totalNorm;
      prob4=norm4/totalNorm;
      prob5=norm5/totalNorm;
      prob6=norm6/totalNorm;
      //wvFns[i]=wvFn;
      wvFn.zeros();
      //wvFn.print("wvFn:");
      //printf("\n%lg\n",prob4);
      randDOrS = drand48();
      sDecay = true;
      randDir = drand48();
      if(randDOrS<(decayRatioD5Halves/(decayRatioD5Halves+1))){//odds that a given decay would be into the D state
	sDecay=false;
	//decaysD=decaysD+1;
	if(randDir<0.5){//theoretically the kick should be in a random direction...but we're not really cold enough for it to matter so I just put it along either plus or minus "x" for simplicity
	  kick=vKickDP;//kick for 1033 photon
	}
	else{
	  kick=-vKickDP;
	}
      }
      else{
	//decaysS=decaysS+1;
	if(randDir<0.5){//if instead decay is to S state, give the kick corresponding to a 408 photon
	  kick=vKick;
	}
	else{
	  kick=-vKick;
	}
      }
      
      if (rand2<prob3)//if ion was found in state 3 (mJ=+3/2)
	{
	  if(sDecay){//only "S" state that 3 can decay to is 2 (mJ=+1/2) (wvFn(1,0) = state 2...c++ is zero indexed!)
	    wvFn(1,0).real(1);
	  }
	  else{//can decay to either +5/2 (12) +3/2(11) or +1/2(10) state.  Roll for it and decide based on C-G coeffs for each possibility (see TKL thesis)
	    rand3 = drand48();
	    if(rand3<gs[17]*gs[17]/decayRatioD5Halves){
	      wvFn(11,0).real(1);
	    }
	    else if(rand3<gs[17]*gs[17]/decayRatioD5Halves+gs[16]*gs[16]/decayRatioD5Halves){
	      wvFn(10,0).real(1);
	    }
	    else{
	      wvFn(9,0).real(1);
	    }
	  }
	  
	}
      else if(rand2<prob3+prob4)//if ion in state 4
	{
	  if(sDecay){//can decay to 1 or 2, roll for probability
	    rand3=drand48();
	    if(rand3<gs[2]*gs[2]){
	      wvFn(0,0).real(1);
	    }
	    else{
	      wvFn(1,0).real(1);
	    }
	  }
	  else{//if d state decay: can be to either 9 10 or 11
	    rand3=drand48();
	    if(rand3<gs[14]*gs[14]/decayRatioD5Halves){
	      wvFn(10,0).real(1);
	    }
	    else if(rand3<gs[14]*gs[14]/decayRatioD5Halves+gs[13]*gs[13]/decayRatioD5Halves){
	      wvFn(9,0).real(1);
	    }
	    else{
	      wvFn(8,0).real(1);
	    }
	    
	  }
	}
      else if(rand2<prob3+prob4+prob5)//if in state 5
	{
	  if(sDecay){//can decay to 1 or 2, roll for probability
	    rand3=drand48();
	    if(rand3<gs[4]*gs[4]){
	      wvFn(1,0).real(1);
	    }
	    else{
	      wvFn(0,0).real(1);
	    }
	  }
	  else{//if d state decay: can be to either 8 9 or 10
	    rand3=drand48();
	    if(rand3<gs[11]*gs[11]/decayRatioD5Halves){
	      wvFn(9,0).real(1);
	    }
	    else if(rand3<gs[11]*gs[11]/decayRatioD5Halves+gs[10]*gs[10]/decayRatioD5Halves){
	      wvFn(8,0).real(1);
	    }
	    else{
	      wvFn(7,0).real(1);
	    }
	  }
	}
      else//if in state 6
	{
	  if(sDecay){//can only decay to 1
	    wvFn(0,0).real(1);
	  }
	  else{//d state decay can be into 7 8 or 9
	    rand3=drand48();
	    if(rand3<gs[8]*gs[8]/decayRatioD5Halves){
	      wvFn(8,0).real(1);
	    }
	    else if(rand3<gs[8]*gs[8]/decayRatioD5Halves+gs[7]*gs[7]/decayRatioD5Halves){
	      wvFn(7,0).real(1);
	    }
	    else{
	      wvFn(6,0).real(1);
	    }
	  }
	}
      //wvFns[i]=wvFn;
      //double randPhaseNum = drand48();
      //randPhase[i]=randPhaseNum*2*3.1419;//comment out for no random phase
      
      
    }
    wvFns[i]=wvFn;
    V[0][i]=V[0][i]+kick;
    if(reNormalizewvFns){
      popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0));
      popP = std::norm(wvFn(2,0)) + std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0));
      popD = std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0)) + std::norm(wvFn(10,0)) + std::norm(wvFn(11,0));
      wvFn = wvFn/sqrt(popS+popP+popD);
      wvFns[i] = wvFn;
    }
    
  }//for all particles, evolve wave function
  }//pragmaOmp
  t+=dtQuant;//step system time
}

/***********************************************************************/
/*                                                                     */
/*  output results / annealed conditions							   */
/*                                                                     */
/***********************************************************************/

void writeConditions(int c0)
{
	FILE *fa;
	char dataDirCopy[256];
	char buffer[256];
	char fileName[256];

	// print file "ions", contains the number of ions, and the counter for vel_dist data
	strcpy(dataDirCopy,saveDirectory);
	sprintf(buffer,"ions_timestep%06d.dat",c0);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"w");
	fprintf(fa,"%i\t%i",N,counter);
	fclose(fa);

	// "conditions" contains all position and velocity data for all particles
  	sprintf(buffer,"conditions_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"w");
  	for (int i = 0; i < N; i++)
  	{
    	fprintf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t\n",R[0][i],R[1][i],R[2][i],V[0][i],V[1][i],V[2][i]);
    }
  	fclose(fa);

	// "VZERO" has the initial velocity data at tstartV0, tstartV1 for VAF calculations
  	for (int c2V = 0; c2V < numberOfIntervalV; c2V++)
    {
    	sprintf(buffer,"VZERO_timestep%06d_interval%d.dat",c0,c2V);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"w");
		for (int i = 0; i < N; i++)
		{
			fprintf(fa,"%lg\t%lg\t%lg\n",Vholder[0][c2V][i],Vholder[1][c2V][i],Vholder[2][c2V][i]);
		}		       			   
    	fclose(fa);
    }
    

	//wvFns contains the wavefunctions of every particle

  	sprintf(buffer,"wvFns_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"w");
	for(int j=0;j<N;j++)
	  {
	    cx_mat currWvFn=wvFns[j];
	    for (int k=0;k<numStates;k++){
	      
	      fprintf(fa,"%lg\t%lg\t",currWvFn(k,0).real(),currWvFn(k,0).imag());
	    }
	    fprintf(fa,"\n");
	  }
	fclose(fa);

      
}
void readConditions(int c0)
{
	lDeb=1./sqrt(3.*Ge);                           // Debye length in units of a
	L=pow(N0*4.*M_PI/3.,0.333333333);              // box length for N0 ions
	t = ((double)c0-9.)*TIMESTEP + 0.02;
  
  
	for(int i=0;i<2001;i++)          // set up bins for velocity distribution
    {                            	// bin size is chosen as 0.0025 and range [-5:5]
    	vel[i]=(double)i*0.0025;
    }
	FILE *fa;
	char dataDirCopy[256];
	char buffer[256];
	char fileName[256];
  	int i = 0;						// iterator for reading from files
  	double a, b, z, d, e, f,ar,ai,br,bi,cr,ci,dr,di,er,ei,fr,fi,gr,gi,hr,hi,ir,ii,jr,ji,kr,ki,lr,li;		// a bunch of dummy variables for reading data files
  	int g, h, k, j, m;
  
  	// Read from "ions"
	strcpy(dataDirCopy,saveDirectory);
	sprintf(buffer,"ions_timestep%06d.dat",c0);
	strcpy(fileName,strcat(dataDirCopy,buffer));
  	fa = fopen(fileName,"r");
  	while (fscanf(fa,"%i\t%i",&j,&m) == 2)
    {
    	N = j;
    	counter = m;
    }
  	fclose(fa);
  
  	i = 0;
  	// Read positions and velocities from "conditions"
	sprintf(buffer,"conditions_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"r");
  	while (fscanf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",&a,&b,&z,&d,&e,&f) == 6)
    {
    	R[0][i] = a;
    	R[1][i] = b;
    	R[2][i] = z;
    	V[0][i] = d;
    	V[1][i] = e;
    	V[2][i] = f;
    	i++;
    }
  	fclose(fa);
	/*
	for(j=0;j<i;j++)              // loop over particles in large box
	  {
	    
	    
	    double rand1 = drand48();
	    double rand2 = drand48();
	    double rand3=drand48();
	    double sign=1;
	    if(rand3<0.5){
	      sign =-1;
		}
	    double rand4=drand48();
	    double sign2=1;
	    if(rand4<0.5){
	      sign2=-1;
		}
	    mat wvFn1Cont = sqrt(rand1)*ident.col(0);
	    mat wvFn2RealCont = sign2*sqrt(1-rand1)*sqrt(rand2)*ident.col(1);
	    mat wvFn2ImCont = sign*sqrt(1-rand1)*sqrt(1-rand2)*ident.col(1);
	    wvFns[j]=cx_mat(wvFn1Cont+wvFn2RealCont,wvFn2ImCont);
	    
	  }
	*/
  	i = 0;
  	// Read wvFns from "conditions"
	sprintf(buffer,"wvFns_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"r");
  	while (fscanf(fa,"%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\t%lg%lg\n",&ar,&ai,&br,&bi,&cr,&ci,&dr,&di,&er,&ei,&fr,&fi,&gr,&gi,&hr,&hi,&ir,&ii,&jr,&ji,&kr,&ki,&lr,&li) == numStates*2)
    {
      cx_mat currWvFn=cx_mat(mat(12,1,fill::zeros),mat(12,1,fill::zeros));
      currWvFn(0,0).real(ar);
      currWvFn(0,0).imag(ai);
      currWvFn(1,0).real(br);
      currWvFn(1,0).imag(bi);
      currWvFn(2,0).real(cr);
      currWvFn(2,0).imag(ci);
      currWvFn(3,0).real(dr);
      currWvFn(3,0).imag(di);
      currWvFn(4,0).real(er);
      currWvFn(4,0).imag(ei);
      currWvFn(5,0).real(fr);
      currWvFn(5,0).imag(fi);
      currWvFn(6,0).real(gr);
      currWvFn(6,0).imag(gi);
      currWvFn(7,0).real(hr);
      currWvFn(7,0).imag(hi);
      currWvFn(8,0).real(ir);
      currWvFn(8,0).imag(ii);
      currWvFn(9,0).real(jr);
      currWvFn(9,0).imag(ji);
      currWvFn(10,0).real(kr);
      currWvFn(10,0).imag(ki);
      currWvFn(11,0).real(lr);
      currWvFn(11,0).imag(li);
      wvFns[i]=currWvFn;

    	i++;
    }
  	fclose(fa);
	
	
  	i = 0;
  	for (int c2V = 0; c2V < numberOfIntervalV; c2V++)
    {
    	i = 0;
	sprintf(buffer,"VZERO_timestep%06d_interval%d.dat",c0,c2V);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"r");
    	while (fscanf(fa,"%lg\t%lg\t%lg",&a,&b,&z) == 3)
		{
	  		Vholder[0][c2V][i] = a;
	  		Vholder[1][c2V][i] = b;
	  		Vholder[2][c2V][i] = z;
	  		i++;
		}
      	fclose(fa);
    }
    
  		
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
    EkinY=0.0;
    EkinZ=0.0;
    //get velX average...necessary for calculating total KE in moving frame.
    for(i=0;i<N;i++)
      {
	velXAvg+=V[0][i];
      }
    velXAvg/=(double)N;//normalize by ion number;
    for(i=0;i<N;i++)   // calculate total kinetic energy (separate x from y,z)
    {
      EkinX+=0.5*((V[0][i]-velXAvg)*(V[0][i]-velXAvg));
      EkinY+=0.5*(V[1][i]*V[1][i]);
      EkinZ+=0.5*(V[2][i]*V[2][i]);
    }
    EkinX/=(double)N;   // normalize by ion number
    EkinY/=(double)N;   // normalize by ion number
    EkinZ/=(double)N;   // normalize by ion number
    Epotential();      // calculate potential energy
    
    // output energies (2. & 3. column) and energy change (4. column) and avg Vel;
    strcpy(dataDirCopy,saveDirectory);
    strcpy(fileName,strcat(dataDirCopy,"energies.dat"));
    fa = fopen(fileName,"a");
    fprintf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",t,EkinX,EkinY,EkinZ,Epot,EkinX+EkinY+EkinZ+Epot-Epot0,velXAvg);
    fclose(fa);
    
    // calculate velocity distribution using gaussian weight functions of width 0.002
    V2=1./(2.*0.002*0.002);
    for(i=0;i<2001;i++)
    {
        PvelX[i]=0.0;
	PvelY[i]=0.0;
	PvelZ[i]=0.0;
    }
    for(i=0;i<N;i++) // add weight functions exploiting isotropy of Pvel (don't do that anymore...separate "x" from y,z)
    {
        for(j=0;j<2001;j++)
        {
	  PvelX[j]+=exp(-V2*(vel[j]-(V[0][i]-velXAvg))*(vel[j]-(V[0][i]-velXAvg)))+exp(-V2*(vel[j]+(V[0][i]-velXAvg))*(vel[j]+(V[0][i]-velXAvg)));
	  PvelY[j]+=exp(-V2*(vel[j]-V[1][i])*(vel[j]-V[1][i]))+exp(-V2*(vel[j]+V[1][i])*(vel[j]+V[1][i]));
	  PvelZ[j]+=exp(-V2*(vel[j]-V[2][i])*(vel[j]-V[2][i]))+exp(-V2*(vel[j]+V[2][i])*(vel[j]+V[2][i]));
        }
    }
    for(i=0;i<2001;i++) // normalize Pvel
    {
        PvelX[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
	PvelY[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
	PvelZ[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
    }
    
    // output distribution

    sprintf(buffer,"vel_distX_time%06d.dat",counter);
    strcpy(dataDirCopy,saveDirectory);
    strcpy(fileName,strcat(dataDirCopy,buffer));
    fa = fopen(fileName,"w");
    
    sprintf(buffer,"vel_distY_time%06d.dat",counter);
    strcpy(dataDirCopy,saveDirectory);
    strcpy(fileName,strcat(dataDirCopy,buffer));
    fa2 = fopen(fileName,"w");
    
    sprintf(buffer,"vel_distZ_time%06d.dat",counter);
    strcpy(dataDirCopy,saveDirectory);
    strcpy(fileName,strcat(dataDirCopy,buffer));
    fa3 = fopen(fileName,"w");
    
    for(i=0;i<2001;i++)
    {
        fprintf(fa,"%lg\t%lg\n",vel[i]+velXAvg,PvelX[i]);
	fprintf(fa2,"%lg\t%lg\n",vel[i],PvelY[i]);
	fprintf(fa3,"%lg\t%lg\n",vel[i],PvelZ[i]);
    }
    fclose(fa);
    fclose(fa2);
    fclose(fa3);
    

    //output ion states vs vel (column1 vx, column2 total S state, column 3 total P state, etc.)
    double currVel,popS,popP,popD;
    cx_mat currWvFn;
    strcpy(dataDirCopy,saveDirectory);
    snprintf(buffer, 256,"statePopulationsVsVTime%06d.dat", counter);
    strcpy(fileName,strcat(dataDirCopy,buffer));
    fa=fopen(fileName, "w");
    for (i=0;i<N;i++){
      currVel=V[0][i];
      currWvFn = wvFns[i];
      popS = std::norm(currWvFn(0,0)) +std::norm(currWvFn(1,0));
      popP = std::norm(currWvFn(2,0)) + std::norm(currWvFn(3,0)) + std::norm(currWvFn(4,0)) + std::norm(currWvFn(5,0));
      popD = std::norm(currWvFn(6,0)) + std::norm(currWvFn(7,0)) + std::norm(currWvFn(8,0)) + std::norm(currWvFn(9,0)) + std::norm(currWvFn(10,0)) + std::norm(currWvFn(11,0));
      fprintf(fa,"%lg\t%lg\t%lg\t%lg\n",currVel,popS,popP,popD);
    }
    fclose(fa);
    
    // advance timestep counter
    counter++;
    
    /**********************************/
    /* ADD MORE OBSERVABLES TO OUTPUT */
    /**********************************/
}

/***********************************************************************/
/*                                                                     */
/*  LONGITUDINAL CURRENT CORRELATION                                   */
/*                                                                     */
/***********************************************************************/

void LCCF(void)
{
	int kcount = 0;
	double vectorkx;
	double vectorky;
	double vectorkz;
	for (int kx = 0; kx < lambdaFRAC; kx++)                                            // average over many k that are roughly the same magnitude
    {
      	for (int ky = 0; ky < lambdaFRAC; ky++)
		{
	  		for (int kz = 0; kz < lambdaFRAC; kz++)
	    	{
	    		J[0][kx][ky][kz] = 0.;
	    		J[1][kx][ky][kz] = 0.;
	    		J[2][kx][ky][kz] = 0.;

		  		vectorkx = 2.0*M_PI*(double)kx/L;
		  		vectorky = 2.0*M_PI*(double)ky/L;
		  		vectorkz = 2.0*M_PI*(double)kz/L;
		  		kcount++;
		  		for (int j = 0; j < N; j++)                                             // calculate fourier transformed current at each timestep within the interval
		    	{
		      		J[0][kx][ky][kz] += V[0][j]*(cos(vectorkx*(R[0][j])+vectorky*(R[1][j])+vectorkz*(R[2][j]))+I*sin(vectorkx*(R[0][j])+vectorky*(R[1][j])+vectorkz*(R[2][j])));
		      		J[1][kx][ky][kz] += V[1][j]*(cos(vectorkx*(R[0][j])+vectorky*(R[1][j])+vectorkz*(R[2][j]))+I*sin(vectorkx*(R[0][j])+vectorky*(R[1][j])+vectorkz*(R[2][j])));
		      		J[2][kx][ky][kz] += V[2][j]*(cos(vectorkx*(R[0][j])+vectorky*(R[1][j])+vectorkz*(R[2][j]))+I*sin(vectorkx*(R[0][j])+vectorky*(R[1][j])+vectorkz*(R[2][j])));
		    	}
	    	}
		}
	}
}
void printJ(double timer, int c1C, int c2C)
{
        FILE *fa;
	char dataDirCopy[256];
	char buffer[256];
	char fileName[256];

	sprintf(buffer,"J_interval%i.dat",c2C);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"a");
	for (int kx = 0; kx < lambdaFRAC; kx++)
	{
    	for (int ky = 0; ky < lambdaFRAC; ky++)
		{
	  		for (int kz = 0; kz < lambdaFRAC; kz++)
	    	{
  				fprintf(fa,"%i\t%i\t%i\t%i\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",c1C,kx,ky,kz,real(J[0][kx][ky][kz]),imag(J[0][kx][ky][kz]),real(J[1][kx][ky][kz]),imag(J[1][kx][ky][kz]),real(J[2][kx][ky][kz]),imag(J[2][kx][ky][kz]));
  			}
  		}
  	}
  	fclose(fa);
}

/***********************************************************************/
/*                                                                     */
/*                VELOCITY AUTOCORRELATION                             */
/*                                                                     */
/***********************************************************************/

void Zfunc(int c1V, int c2V)
{
	if (c1V == 0)								// if you are at start of interval..
	{
		for (int j = 0; j < N; j++)             // Save initial velocity for this time interval
    	{
    		Vholder[0][c2V][j] = V[0][j];
    		Vholder[1][c2V][j] = V[1][j];
    		Vholder[2][c2V][j] = V[2][j];
    	}
    }
    VAF = 0.0;
    for(int j = 0; j<N ; j++)                           
	{                                          
	  	VAF += 1/((double)(N)) * (Vholder[0][c2V][j]*V[0][j]+Vholder[1][c2V][j]*V[1][j]+Vholder[2][c2V][j]*V[2][j]);
	  	// Calculate velocity autocorrelation function at this time
	}				
}
void printVAF(double time, int c1V, int c2V)
{
        FILE *fa;
	char dataDirCopy[256];
	char buffer[256];
	char fileName[256];
	sprintf(buffer,"VAF_interval%i.dat",c2V);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa = fopen(fileName,"a");
	fprintf(fa,"%lg\t%lg\n",time,VAF);
	fclose(fa);
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
  char namebuf[256];
  char namebuf2[256];
  char saveDirBackup[256];
  //strcpy(saveDirBackup,saveDirectory);
  sprintf(namebuf,"Ge%dDensity%dE+11Sig0%dTe%dSigFrac%dDetSP%dDetDP%dOmSP%dOmDP%dNumIons%d",(unsigned)(100*Ge),(unsigned)(density*1000),(unsigned)(10*sig0),(unsigned)(Te),(unsigned)(fracOfSig*100),(unsigned)(detuning*100),(unsigned)(detuningDP*100),(unsigned)(Om*100),(unsigned)(OmDP*100),(unsigned)N0);
  strcat(saveDirectory,namebuf);
  mkdir(saveDirectory,ACCESSPERMS);
  //make directory for given job
  sprintf(namebuf2,"/job%d/",job);
  strcat(saveDirectory,namebuf2);
  mkdir(saveDirectory,ACCESSPERMS);
  //saveDirectory is now of form "OriginalSaveDirectory/Gamma%d...etc/job1/"
  
  //have to define all this here for some reason instead of the global varaible section...
  cs[0] = wvFn2*wvFn3.t();
  cs[1] = wvFn2*wvFn4.t();
  cs[2] = wvFn1*wvFn4.t();
  cs[3] = wvFn1*wvFn5.t();
  cs[4] = wvFn2*wvFn5.t();
  cs[5] = wvFn1*wvFn6.t();
  cs[6] = wvFn7*wvFn6.t();
  cs[7] = wvFn8*wvFn6.t();
  cs[8] = wvFn9*wvFn6.t();
  cs[9] = wvFn8*wvFn5.t();
  cs[10] = wvFn9*wvFn5.t();
  cs[11] = wvFn10*wvFn5.t();
  cs[12] = wvFn9*wvFn4.t();
  cs[13] = wvFn10*wvFn4.t();
  cs[14] = wvFn11*wvFn4.t();
  cs[15] = wvFn10*wvFn3.t();
  cs[16] = wvFn11*wvFn3.t();
  cs[17] = wvFn12*wvFn3.t();
  gs[0]=sqrt(1.);
  gs[1]=sqrt(2./3);
  gs[2]=sqrt(1./3);
  gs[3]=sqrt(2./3);
  gs[4]=sqrt(1./3);
  gs[5]=sqrt(1.);
  gs[6]=sqrt(decayRatioD5Halves*2./3);
  gs[7]=sqrt(decayRatioD5Halves*4./15);
  gs[8]=sqrt(decayRatioD5Halves*1./15);
  gs[9]=sqrt(decayRatioD5Halves*2./5);
  gs[10]=sqrt(decayRatioD5Halves*2./5);
  gs[11]=sqrt(decayRatioD5Halves*1./5);
  gs[12]=sqrt(decayRatioD5Halves*1./5);
  gs[13]=sqrt(decayRatioD5Halves*2./5);
  gs[14]=sqrt(decayRatioD5Halves*2./5);
  gs[15]=sqrt(decayRatioD5Halves*1./15);
  gs[16]=sqrt(decayRatioD5Halves*4./15);
  gs[17]=sqrt(decayRatioD5Halves*2./3);


  for(int j=0;j<18;j++){
    hamDecayTerm=hamDecayTerm-1./2*I*(gs[j]*gs[j]*cs[j].t()*cs[j]);
    decayMatrix = decayMatrix + gs[j]*gs[j]*cs[j].t()*cs[j];
  }
  
  for(int k=0;k<6;k++){
    if(k!=1 && k!=3){
      hamCouplingTermNoTimeDep += -1.*cs[k].t()*gs[k]*Om/2;
    }
  }
  for(int k=6;k<18;k++){
    if(k!=8 && k!=11 && k!=7 && k!=10 && k!= 13 && k!=16){
      hamCouplingTermNoTimeDep += -1.*cs[k].t()*gs[k]*OmDP/2/sqrt(decayRatioD5Halves);
    }
  }
    
  

	srand48((unsigned)time(NULL)+job); // initialize random number generator

	int cstart0 = (int)((tstartC0-0.02)/TIMESTEP + 9);      //start timestep of each interval
	int vstart0 = (int)((tstartV0-0.02)/TIMESTEP + 9);
	int vstart1 = (int)((tstartV1-0.02)/TIMESTEP + 9);
	int vstart2 = (int)((tstartV2-0.02)/TIMESTEP + 9);
	int vstart3 = (int)((tstartV3-0.02)/TIMESTEP + 9);
	int vstart4 = (int)((tstartV4-0.02)/TIMESTEP + 9);
	int vstart5 = (int)((tstartV5-0.02)/TIMESTEP + 9);
	int vstart6 = (int)((tstartV6-0.02)/TIMESTEP + 9);
	int vstart7 = (int)((tstartV7-0.02)/TIMESTEP + 9);
	int vstart8 = (int)((tstartV8-0.02)/TIMESTEP + 9);
	int vstart9 = (int)((tstartV9-0.02)/TIMESTEP + 9);
	int vstart10 = (int)((tstartV10-0.02)/TIMESTEP + 9);
	int vstart11 = (int)((tstartV11-0.02)/TIMESTEP + 9);
	int vstart12 = (int)((tstartV12-0.02)/TIMESTEP + 9);
	int timeStepCounter =plasmaToQuantumTimestepRatio;  //how many quantum timesteps have we undergone?  every so often, do plasma stuff
  
  	// if it's a new run, initialize in normal way, otherwise read from annealed data files
  	if (newRun == 1)
  	{
  		init();
  	}
  	if (newRun == 0)
  	{
		readConditions(c0);
  	}
  
  	
  	while(t<=tmax+0.0009)                     // run simulation until tmax
	{
		// Calculation of CCF
	  /*
		if(c0 >= cstart0 && c0 < (cstart0 + lengthOfIntervalC) && (c0-cstart0)%sampleFreq == 0)
	    {
	      LCCF();
			printJ(t,c0-cstart0,0);
	    }

		// Calculation of VAF
		// interval 0
	  	if(c0 >= vstart0  && c0 < (vstart0 + lengthOfIntervalV) && (c0-vstart0)%sampleFreq == 0)
	    {
	    	
	    	
			Zfunc(c0-vstart0,0);
	    	printVAF(t,c0-vstart0,0);
	    }
	  	// interval 1
	  	if(c0 >= vstart1 && c0 < (vstart1 + lengthOfIntervalV) && (c0-vstart1)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart1,1); 
	    	printVAF(t,c0-vstart1,1);
	    }
	  	// interval 2
	  	if(c0 >= vstart2 && c0 < (vstart2 + lengthOfIntervalV) && (c0-vstart2)%sampleFreq == 0)
	    {
	    
	    	
	    	Zfunc(c0-vstart2,2); 
	    	printVAF(t,c0-vstart2,2);
	    }
	    // interval 3
	  	if(c0 >= vstart3 && c0 < (vstart3 + lengthOfIntervalV) && (c0-vstart3)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart3,3); 
	    	printVAF(t,c0-vstart3,3);
	    }
	    // interval 4
	  	if(c0 >= vstart4 && c0 < (vstart4 + lengthOfIntervalV) && (c0-vstart4)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart4,4); 
	    	printVAF(t,c0-vstart4,4);
	    }
	    // interval 5
	  	if(c0 >= vstart5 && c0 < (vstart5 + lengthOfIntervalV) && (c0-vstart5)%sampleFreq == 0)
	    {
	    
	    	
	    	Zfunc(c0-vstart5,5); 
	    	printVAF(t,c0-vstart5,5);
	    }
	    // interval 6
	  	if(c0 >= vstart6 && c0 < (vstart6 + lengthOfIntervalV) && (c0-vstart6)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart6,6); 
	    	printVAF(t,c0-vstart6,6);
	    }
	    // interval 7
	  	if(c0 >= vstart7 && c0 < (vstart7 + lengthOfIntervalV) && (c0-vstart7)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart7,7); 
	    	printVAF(t,c0-vstart7,7);
	    }
	    // interval 8
	  	if(c0 >= vstart8 && c0 < (vstart8 + lengthOfIntervalV) && (c0-vstart8)%sampleFreq == 0)
	    {
	    
	    	
	    	Zfunc(c0-vstart8,8); 
	    	printVAF(t,c0-vstart8,8);
	    }
	    // interval 9
	  	if(c0 >= vstart9 && c0 < (vstart9 + lengthOfIntervalV) && (c0-vstart9)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart9,9); 
	    	printVAF(t,c0-vstart9,9);
	    }
	    // interval 10
	  	if(c0 >= vstart10 && c0 < (vstart10 + lengthOfIntervalV) && (c0-vstart10)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart10,10); 
	    	printVAF(t,c0-vstart10,10);
	    }
	    // interval 11
	  	if(c0 >= vstart11 && c0 < (vstart11 + lengthOfIntervalV) && (c0-vstart11)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart11,11); 
	    	printVAF(t,c0-vstart11,11);
	    }
	    // interval 12
	  	if(c0 >= vstart12 && c0 < (vstart12 + lengthOfIntervalV) && (c0-vstart12)%sampleFreq == 0)
	    {
	    	
	    	
	    	Zfunc(c0-vstart12,12); 
	    	printVAF(t,c0-vstart12,12);
		}*/

		// other outputs like energy, vel_dist
	    if ((c0+1)%sampleFreq == 0&&timeStepCounter==1)
	    {
	    	output();
	    }
	    if(timeStepCounter==plasmaToQuantumTimestepRatio){
	      //step();(Now we just redo the MD forces every "full" MD timestep.  step() now applies that force divided by the "ratio"  for every quantum timestep.
	      //THus, we make the same number of force calculations, but the velocity and position steps are "parcelled out" in finer steps, so the QT code doesn't see giant velocity changes.
	      forces();
	      c0++;
	      timeStepCounter=0;
	    }
	    step();
	    qstep();
	    timeStepCounter++;
	}
  	//Print conditions
  	writeConditions(c0);
  	return 0;
}
