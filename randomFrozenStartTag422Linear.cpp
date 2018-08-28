/***********************************************************************/
/*                                                                     */
/*  MD simulation of ions interacting via screened Coulomb forces      */
/*  initial conditions correspond to zero temperature and flat density */
/*  but can be changed to accomodate finite velocities                 */
/*  and initial density perturbations                                  */
/*                                                                     */
/*  We have also added the ability to simulate the optical pumping     */
/*  done by cross cicrularly polarized counter propagating             */
/*  red detuned 422 lasers, as in our PRL and PRX papers.              */
/*  After simulating the evolution of each ion's wavefunction          */
/*  for a user selected pump time, particles are tagged with           */
/*  a probability <\uparrow|\psi> and the velocity of these            */
/*  particles is recorded subsequently.  The VAF is also recorded      */
/*  starting at this time.  If the pumping is sufficiently linear,     */
/*  the two should match                                               */
/*                                                                     */
/***********************************************************************/


/******************************************************************************************************************************************/
/*                                                                                                                                        */
/*    To Compile: first at some point you must load the g++ compiler and c++11 by typing "module load GCC/4.9.3" Then, to compile, type   */
/*                                                                                                                                        */
/*    g++ -std=c++11 -fopenmp -o tag422 -O3 simOptPumping422.cpp -lm -I/home/tkl1/usr/include -L/users/tkl1/user/lib64                    */
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
#include<cstring>

using namespace std;
using namespace arma;

/* GLOBAL VARIABLES */

/*directory where files will be saved*/

char saveDirectory[256] = "data422/";//main directory.  A subfolder titled  pumpTime___PumpStart___Om___Det___Density__Ge__ will be created.  Subsubfolders for each job will be created.  data will be stored in the job folders;

//pumping parameters
double detuning=-1;//pump detuning in units \gamma_{422}
double Om=1.3;//rabi frequency in units \gamma_{422}
double tpumpreal=0.0000001;// in seconds

/* input variables */
int nrthread = 4;              // number of threads
double Ge = 0.1;			   // kappa = sqrt(3*Ge)
int newRun = 1;			       // Is it a new run (1), or is it starting from annealed conditions (0)?
int c0 = 0;				   // timestep counter, if starting from annealed conditions, this should be the same as the timestep label on the input files
int sampleFreq = 40;		   // output data for all functions every X timesteps
double density = 2;              //units of 10^14 m^-3
double gamToEinsteinFreq = 174.07*.894/sqrt(density);// ratio of gamma=1.26e8 to einstein freq
#define N0 3500               // average particle number in simulation cell
#define TIMESTEP 0.002		   // default time step
#define tmax 25                 // maximum simulation time
#define lambdaFRAC 12           // 1+max value of a single integer in the k integer triplet
#define tstartC0 0.88          // MUST BE GREATER THAN 0.02
int plasmaToQuantumTimestepRatio = (int) round(34.81*.894/sqrt(density));//1/5 of above quantity because MD timestep is 0.002omega-1 while quantum timestep of 0.01gamma-1
double quantumTimestep = TIMESTEP/plasmaToQuantumTimestepRatio;//timestep for quantum evolution (0.01Gam) in plasma time units
double plasVelToQuantVel=1.1821*pow(density,1./6)*.967;//conversion factor for going from plasma to quantum velocities (norm by a\omega and k/\gamma respectively)

							   // Only need one CCF interval, since it outputs J
#define tstartV0 15			   // need to change main() if you want to add intervals
double tpump = tpumpreal*813490*sqrt(density);//in omega_{E}t
double tendV0=tstartV0+tpump;
int recordedSpinUps=0;


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
double PvelX[4001];          // velocity distribution
double PvelY[4001];          // velocity distribution
double PvelZ[4001];          // velocity distribution
double vel[4001];           // corresponding velocity bins

/* system variables */
double R[3][N0+1000];        // ion positions
double V[3][N0+1000];        // ion velocities
double F[3][N0+1000];        // forces for each ion
int SpinUpList[N0+1000];     //whetehr or not ion is spin up
int NSpinUp;
unsigned N;             // number of particles

/* kinetic longitudinal stress Autocorrelation Function */
double VAF;     								     // velocity autocorrelation function
double Vholder[N0+1000];       // Hold Velocity data

std::complex<double> I(0,1);


/*QUANTUM STUFF*/
double decayRatio=0.0754;//ratio of D decay to S decay (9.5)/(126), see NIST spectral data
double vKick = 0.001257/plasVelToQuantVel;//vKick=\hbar*k/m in plasma units (it's 0.001257 in quantum units)

//waveFunctions
cx_mat wvFns[N0+1000];
double numStates = 5;
mat ident = mat(numStates,numStates,fill::eye);
cx_mat wvFn1=cx_mat(ident.col(0),mat(numStates,1,fill::zeros));//S, mJ=-1/2
cx_mat wvFn2=cx_mat(ident.col(1),mat(numStates,1,fill::zeros));//S, mJ=+1/2
cx_mat wvFn3=cx_mat(ident.col(2),mat(numStates,1,fill::zeros));//P, mJ=+1/2
cx_mat wvFn4=cx_mat(ident.col(3),mat(numStates,1,fill::zeros));//P, mJ=-1/2
cx_mat wvFn5=cx_mat(ident.col(4),mat(numStates,1,fill::zeros));//D state
//decay coupling and rates
cx_mat cs[6];
double gs[6];


/* FUNCTIONS */
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

            N++;                    // increase particle number by 1
        }
    }
    printf("%i\n",N);
    
    for(i=0;i<4001;i++)           // set up bins for velocity distribution
    {                            // bin size is chosen as 0.0025 and range [-5:5]
      vel[i]=(double)(i-2000)*0.0025;

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
        forces();
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
    
    forces();
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
  dt=quantumTimestep*plasmaToQuantumTimestepRatio;
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

  //422 pumping, no kick
  unsigned i;
  cx_mat wvFn;
  double velQuant;
  double velPlas;
  double tPart;
  cx_mat dpmatTerms[6];
  mat zero_mat1=mat(1,1,fill::zeros);
  double dtQuant = quantumTimestep;
  double kick;
  cx_mat densMatrix;
  //hamiltonian and various terms
  double totalDetRightSP;
  double totalDetLeftSP;
  cx_mat hamCouplingTermSP;
  cx_mat hamCouplingTermDP;
  cx_mat hamEnergyTermP;
  cx_mat hamEnergyTermD;
  cx_mat hamDecayTerm;
  
  for(i=0;i<N;i++){
    cx_mat dpmat=cx_mat(zero_mat1,zero_mat1);
    wvFn=wvFns[i];
    velPlas=V[0][i];//use x velocity, lasers going along x dir, velocity in PLASMA units
    velQuant=velPlas*plasVelToQuantVel;//convert velocity to quantum units
    tPart = t*gamToEinsteinFreq;
    for(int j=0;j<6;j++){
      dpmatTerms[j] = dtQuant*gamToEinsteinFreq*wvFn.t()*cs[j].t()*cs[j]*wvFn;
      dpmat=dpmat+dpmatTerms[j]*gs[j];
    }//for all states, calculate dpmat
    double dp = dpmat(0,0).real();
    double rand = drand48();

    if(rand>dp)
      {
	densMatrix = wvFn*wvFn.t();
	//hamiltonian and various terms
	double totalDetRightSP = -detuning-velQuant;//propegating leftward, from right
	double totalDetLeftSP = -detuning+velQuant;
	cx_mat hamCouplingTermSP = -Om/2*wvFn2*wvFn3.t()*sqrt(gs[0])-Om/2*wvFn1*wvFn4.t()*sqrt(gs[2]);
	cx_mat hamEnergyTerm = totalDetRightSP*(wvFn3*wvFn3.t())+totalDetLeftSP*(wvFn4*wvFn4.t());
      
    
	cx_mat hamWithoutDecay=hamEnergyTerm+hamCouplingTermSP+hamCouplingTermSP.t();
	
	hamDecayTerm=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));
	for(int j=0;j<6;j++){
	  hamDecayTerm=hamDecayTerm-1./2*I*(gs[j]*cs[j].t()*cs[j]);
	}
	
	//cx_mat hamDecayTerm(zeros<mat>(numStates,numStates),-1*hamDecayTermComplex);
	
	cx_mat hamil=hamWithoutDecay+hamDecayTerm;
	
	//with hamiltonian calculated, can evolve wvFn using RK method

	double dtHalf = dtQuant*gamToEinsteinFreq/2;
	//get k1,y1 (k1 is slope at t0 calculated using y0, y0 is initial wvFn value.  y1 (wvFnk1) is wvFn stepped by dt/2 w/ slope k1)
	cx_mat dpmatTermsk1[6];
	cx_mat dpmatk1=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<6;j++){
	  dpmatTermsk1[j] = dtQuant*gamToEinsteinFreq*wvFn.t()*cs[j].t()*cs[j]*wvFn;
	  dpmatk1=dpmatk1+dpmatTermsk1[j]*gs[j];
	}
	double dpk1 = dpmatk1(0,0).real();
	double prefactork1 = 1/sqrt(1-dpk1);
	cx_mat matPrefactork1 = ident-I*dtQuant*gamToEinsteinFreq*hamil;
	cx_mat wvFnStepped = prefactork1*matPrefactork1*wvFn;
	cx_mat k1 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFn);
	cx_mat wvFnk1 = wvFn+dtHalf*k1;
	
	//get k2,y2 (k2 is slope at t0+dt/2 calculated using y1, y2 (wvFnk2) is wvFn stepped by dt/2 w/ slope k2)
	cx_mat dpmatTermsk2[6];
	cx_mat dpmatk2=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<6;j++){
	  dpmatTermsk2[j] = dtQuant*gamToEinsteinFreq*wvFnk1.t()*cs[j].t()*cs[j]*wvFnk1;
	  dpmatk2=dpmatk2+dpmatTermsk2[j]*gs[j];
	}
	double dpk2 = dpmatk2(0,0).real();
	double prefactork2 = 1/sqrt(1-dpk2);
	cx_mat matPrefactork2 = ident-I*dtQuant*gamToEinsteinFreq*hamil;
	wvFnStepped = prefactork2*matPrefactork2*wvFnk1;
	cx_mat k2 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFnk1);
	cx_mat wvFnk2 = wvFn+dtHalf*k2;

	
	//get k3, y3 (k3 is slope at t0+dt/2 calculated using y2, y3 (wvFnk3) is wvFn stepped by dt w/ slope k3)
	
	cx_mat dpmatTermsk3[6];
	cx_mat dpmatk3=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<6;j++){
	  dpmatTermsk3[j] = dtQuant*gamToEinsteinFreq*wvFnk2.t()*cs[j].t()*cs[j]*wvFnk2;
	  dpmatk3=dpmatk3+dpmatTermsk3[j]*gs[j];
	}
	double dpk3 = dpmatk3(0,0).real();
	double prefactork3 = 1/sqrt(1-dpk3);
	cx_mat matPrefactork3 = ident-I*dtQuant*gamToEinsteinFreq*hamil;
	wvFnStepped = prefactork3*matPrefactork3*wvFnk2;
	cx_mat k3 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFnk2);
	cx_mat wvFnk3 = wvFn+dtQuant*gamToEinsteinFreq*k3;
	
	//get k4, yfinal (k4 is slope at t0+dt calculated using y3, yfinal is wvFn stepped by dt using weighted average of k1,k2,k3, and k4)
	
	cx_mat dpmatTermsk4[6];
	cx_mat dpmatk4=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<6;j++){
	  dpmatTermsk4[j] = dtQuant*gamToEinsteinFreq*wvFnk3.t()*cs[j].t()*cs[j]*wvFnk3;
	  dpmatk4=dpmatk4+dpmatTermsk4[j]*gs[j];
	}
	double dpk4 = dpmatk4(0,0).real();
	double prefactork4 = 1/sqrt(1-dpk4);
	cx_mat matPrefactork4 = ident-I*dtQuant*gamToEinsteinFreq*hamil;
	wvFnStepped = prefactork4*matPrefactork4*wvFnk3;
	cx_mat k4 = 1./(dtQuant*gamToEinsteinFreq)*(wvFnStepped-wvFnk3);
	wvFn = wvFn+(k1+3*k2+3*k3+k4)/8*(dtQuant*gamToEinsteinFreq);
      }
    else{
      double rand2 = drand48();
      double norm3 = std::norm(wvFn(2,0));
      double norm4 = std::norm(wvFn(3,0));
      double totalNorm = norm3+norm4;
      double prob3=norm3/totalNorm;
      double prob4=norm4/totalNorm;
      //wvFns[i]=wvFn;
      wvFn.zeros();
      //wvFn.print("wvFn:");
      //printf("\n%lg\n",prob4);
      double randDOrS = drand48();
      bool sDecay = true;
      if(randDOrS<(decayRatio/(decayRatio+1))){
	sDecay=false;
      }
      if(rand2<prob3)
	{
	  if(sDecay){
	    double rand3=drand48();
	    if(rand3<gs[0]){
	      wvFn(1,0).real(1);
	    }
	    else{
	      wvFn(0,0).real(1);
	    }
	  }
	  else{
	    wvFn(4,0).real(1);
	  }
	    
	}
      else
	{
	  if(sDecay){
	    double rand3=drand48();
	    if(rand3<gs[2]){
	      wvFn(0,0).real(1);
	    }
	    else{
	      wvFn(1,0).real(1);
	    }
	  }
	  else{
	    wvFn(4,0).real(1);
	  }

	}
      //wvFns[i]=wvFn;
      //double randPhaseNum = drand48();
      //randPhase[i]=randPhaseNum*2*3.1419;//comment out for no random phase
      
      
    }
    wvFns[i]=wvFn;
  }//for all particles, evolve wave function
  t+=dtQuant;//step system time
}

void measureSpinUps(void)
{

  unsigned i;
  cx_mat wvFn;
  int numSpinUp=0;
  for(i=0;i<N;i++){
    wvFn=wvFns[i];
    double rand=drand48();
    double norm1=std::norm(wvFn(0,0));
    double norm2=std::norm(wvFn(1,0));
    double norm3=std::norm(wvFn(2,0));
    double norm4=std::norm(wvFn(3,0));
    if(rand<norm1){
      SpinUpList[i]=1;
      numSpinUp++;
    }
    else if(rand<norm1+norm3){
      double rand2=drand48();
      if(rand2<1./3){
	SpinUpList[i]=1;
	numSpinUp++;
      }
      else{
	SpinUpList[i]=0;
      }
    }
    else if(rand<norm1+norm3+norm4){
      double rand3=drand48();
      if(rand3<2./3){
	SpinUpList[i]=1;
	numSpinUp++;
      }
      else{
	SpinUpList[i]=0;
      }

    }
    else{
      SpinUpList[i]=0;
    }

  }
  NSpinUp=numSpinUp;
  //print file "spinUpIons"

  FILE *fa;
  char namebuf[256];
  char namebuf2[256];
  char dataDirCopy[256];
  sprintf(namebuf,"spinUpIons_timestep%06d.dat",c0);
  strcpy(dataDirCopy,saveDirectory);
  strcpy(namebuf2,strcat(dataDirCopy,namebuf));
  fa = fopen(namebuf2,"w");
  fprintf(fa,"%i",NSpinUp);
  fclose(fa);



}

/***********************************************************************/
/*                                                                     */
/*  output results / annealed conditions							   */
/*                                                                     */
/***********************************************************************/

void writeConditions(int c0)
{
	FILE *fa;
	char namebuf[256];
	char namebuf2[256];
	char namebuf3[256];
	char dataDirCopy[256];

	// print file "ions", contains the number of ions, and the counter for vel_dist data
	sprintf(namebuf,"ions_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf2,strcat(dataDirCopy,namebuf));
	fa = fopen(namebuf2,"w");
	fprintf(fa,"%i\t%i",N,counter);
	fclose(fa);

	//print file "spinUpIonsList", contains list of which ions are spin up
	sprintf(namebuf,"spinUpIonsList_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf2,strcat(dataDirCopy,namebuf));
	fa = fopen(namebuf2,"w");
	for (int i=0;i<N;i++){
	  fprintf(fa,"%i\n",SpinUpList[i]);
	}
	fclose(fa);

	// "conditions" contains all position and velocity data for all particles
  	sprintf(namebuf,"conditions_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf2,strcat(dataDirCopy,namebuf));
  	fa = fopen(namebuf2,"w");
  	for (int i = 0; i < N; i++)
  	{
    	fprintf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t\n",R[0][i],R[1][i],R[2][i],V[0][i],V[1][i],V[2][i]);
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
  	char namebuf[256];
	char namebuf2[256];
	char dataDirCopy[256];
  	int i = 0;						// iterator for reading from files
  	double a, b, z, d, e, f,ar,ai,br,bi,cr,ci,dr,di,er,ei,fr,fi,gr,gi,hr,hi,ir,ii,jr,ji,kr,ki,lr,li;		// a bunch of dummy variables for reading data files
  	int g, h, k, j, m,sUp;
  
  	// Read from "ions"
  	sprintf(namebuf,"ions_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf2,strcat(dataDirCopy,namebuf));
  	fa = fopen(namebuf2,"r");
  	while (fscanf(fa,"%i\t%i",&j,&m) == 2)
    {
    	N = j;
    	counter = m;
    }
  	fclose(fa);

	//Read from "spinUpIonsList"
	i=0;
	sprintf(namebuf,"spinUpIonsList_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf2,strcat(dataDirCopy,namebuf));
	fa=fopen(namebuf2,"r");
	while(fscanf(fa,"%i\n",&sUp)==1)
	  {
	    SpinUpList[i]=sUp;
	    i++;
	  }
  
  	i = 0;
  	// Read positions and velocities from "conditions"
	sprintf(namebuf,"conditions_timestep%06d.dat",c0);
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf2,strcat(dataDirCopy,namebuf));
  	fa = fopen(namebuf2,"r");
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
  
}
void output(void)
{
    unsigned i,j;
    double V2;         // width of gaussian weight function for Pvel(vel)
    double EkinX,EkinY,EkinZ;       // average kinetic energy along each direction
    char namebuf[256];
    char namebuf2[256];
    char namebuf3[256];
    char dataDirCopy[256];
    FILE *fa;
    FILE *fa2;
    FILE *fa3;
    
    EkinX=0.0;
    EkinY=0.0;
    EkinZ=0.0;
    for(i=0;i<N;i++)   // calculate total kinetic energy (separate x from y,z)
    {
      EkinX+=0.5*(V[0][i]*V[0][i]);
      EkinY+=0.5*(V[1][i]*V[1][i]);
      EkinZ+=0.5*(V[2][i]*V[2][i]);
    }
    EkinX/=(double)N;   // normalize by ion number
    EkinY/=(double)N;   // normalize by ion number
    EkinZ/=(double)N;   // normalize by ion number
    Epotential();      // calculate potential energy
    
    // output energies (2. & 3. column) and energy change (4. column); should ideally be zero
    //namebuf = "energies.dat";
    strcpy(dataDirCopy,saveDirectory);
    strcpy(namebuf2,strcat(dataDirCopy,"energies.dat"));
    fa=fopen(namebuf2,"a");
    fprintf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",t,EkinX,EkinY,EkinZ,Epot,EkinX+EkinY+EkinZ+Epot-Epot0);
    fclose(fa);
    
    // calculate velocity distribution using gaussian weight functions of width 0.002 for spin up ions
    //also record average velocity for spin up ions
    V2=1./(2.*0.002*0.002);
    for(i=0;i<4001;i++)
    {
        PvelX[i]=0.0;
	PvelY[i]=0.0;
	PvelZ[i]=0.0;
    }
  
    double firstMom=0,secondMom=0,thirdMom=0,fourthMom=0;
    double currVx;
    unsigned numTagged=0;
    
    for(i=0;i<N;i++) // add weight functions exploiting isotropy of Pvel (don't do that anymore...separate "x" from y,z)
    {
      currVx = V[0][i];
      if(SpinUpList[i])
	{
	  firstMom+=currVx;
	  secondMom+=currVx*currVx;
	  thirdMom+=currVx*currVx*currVx;
	  fourthMom+=currVx*currVx*currVx*currVx;
	  numTagged+=1;
	}
      for(j=0;j<4001;j++)
        {
	  if(SpinUpList[i]==1){
            PvelX[j]+=exp(-V2*(vel[j]-V[0][i])*(vel[j]-V[0][i]));
            PvelY[j]+=exp(-V2*(vel[j]-V[1][i])*(vel[j]-V[1][i]));
            PvelZ[j]+=exp(-V2*(vel[j]-V[2][i])*(vel[j]-V[2][i]));
	  }
        }
    }

    //record subset distro moments
    char fileName[256];
    firstMom/=numTagged;
    secondMom/=numTagged;
    thirdMom/=numTagged;
    fourthMom/=numTagged;
    strcpy(dataDirCopy,saveDirectory);
    strcpy(fileName,strcat(dataDirCopy,"taggedMoments.dat"));
    fa=fopen(fileName, "a");
    fprintf(fa, "%lg\t%lg\t%lg\t%lg\t%lg\n", t, firstMom, secondMom, thirdMom, fourthMom);
    fclose(fa);

    
    for(i=0;i<4001;i++) // normalize Pvel
    {
        PvelX[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
	PvelY[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
	PvelZ[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
    }
    
    // output distribution
    sprintf(namebuf,"vel_distX_timestep%06d.dat",c0);
    strcpy(dataDirCopy,saveDirectory);
    strcpy(namebuf2,strcat(dataDirCopy,namebuf));
    fa=fopen(namebuf2,"w");
    //sprintf(namebuf2,"test/vel_distY_Ge%d_time%06d_job%06d.dat",(unsigned)(1000.*Ge),counter,job);
    //fa2=fopen(namebuf2,"w");
    //sprintf(namebuf3,"test/vel_distZ_Ge%d_time%06d_job%06d.dat",(unsigned)(1000.*Ge),counter,job);
    //fa3=fopen(namebuf3,"w");
    for(i=0;i<4001;i++)
    {
        fprintf(fa,"%lg\t%lg\n",vel[i],PvelX[i]);
	//	fprintf(fa2,"%lg\t%lg\n",vel[i],PvelY[i]);
	//fprintf(fa3,"%lg\t%lg\n",vel[i],PvelZ[i]);
    }
    fclose(fa);
    //fclose(fa2);
    //fclose(fa3);
    
    /*
    //output wvFns

    sprintf(namebuf,"testDIHFullParticles/wvFns_time%06d_Ge%d_job%06d.dat",counter,(unsigned)(1000.*Ge),job);
    fa = fopen(namebuf,"w");
    for(int j=0;j<N;j++)
      {
	cx_mat currWvFn=wvFns[j];
	for (int k=0;k<numStates;k++){
	  
	  fprintf(fa,"%lg+%lgi\t",currWvFn(k,0).real(),currWvFn(k,0).imag());
	    }
	fprintf(fa,"\n");
      }
    fclose(fa);
    */
    // advance timestep counter
    counter++;
    
    /**********************************/
    /* ADD MORE OBSERVABLES TO OUTPUT */
    /**********************************/
}

/***********************************************************************/
/*                                                                     */
/*                VELOCITY AUTOCORRELATION                             */
/*                                                                     */
/***********************************************************************/

void Zfunc(int c1V)
{
  //get mean velocity
  double totalVelSq=0;
  for(int j=0;j<N;j++){
    totalVelSq+=V[0][j]*V[0][j];

  }
  double avgVelSq=totalVelSq/N;

	if (c1V == 0)								// if you are at start of interval..
	{
		for (int j = 0; j < N; j++)             // Save initial velocity for this time interval
    	{
    		Vholder[j] = V[0][j];
    	}
    }
    VAF = 0.0;
    for(int j = 0; j<N ; j++)                           
	{                                          
	  VAF += 1/((double)(N)) * (Vholder[j]*V[0][j]);
	  	// Calculate velocity autocorrelation function at this time
	}				
}

void printVAF(double time)
{
	char namebuf[256];
	char dataDirCopy[256];
	FILE *fa;
	strcpy(dataDirCopy,saveDirectory);
	strcpy(namebuf,strcat(dataDirCopy,"VAF.dat"));
	fa=fopen(namebuf,"a");
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
  job=(unsigned)atof(argv[1]);       // input job label
  //make new main directory
  mkdir(saveDirectory,ACCESSPERMS);
  //make new sub directory
  char namebuf[256];
  char namebuf2[256];
  sprintf(namebuf,"PumpTime%dPumpStart%dDet%dOm%dDensity%dGe%dNumIons%d",(unsigned)(1000000000.*tpumpreal),(unsigned)(tstartV0),(unsigned)(100.*abs(detuning)),(unsigned)(100.*Om),(unsigned) (10.*density),(unsigned)(1000*Ge),(unsigned)N0);
  strcat(saveDirectory,namebuf);
  mkdir(saveDirectory,ACCESSPERMS);
  //make directory for given job
  sprintf(namebuf2,"/job%d/",job);
  strcat(saveDirectory,namebuf2);
  mkdir(saveDirectory,ACCESSPERMS);
  //saveDirectory is now of form "OriginalSaveDirectory/PumpTime%d...etc/job1/"

  //have to define all this here for some reason instead of the global varaible section...
  cs[0] = wvFn2*wvFn3.t();
  cs[1] = wvFn2*wvFn4.t();
  cs[2] = wvFn1*wvFn4.t();
  cs[3] = wvFn1*wvFn3.t();
  cs[4] = wvFn5*wvFn3.t();
  cs[5] = wvFn5*wvFn4.t();
  gs[0]=2./3;
  gs[1]=1./3;
  gs[2]=2./3;
  gs[3]=1./3;
  gs[4]=decayRatio;
  gs[5]=decayRatio;


	
	srand48((unsigned)time(NULL)+job); // initialize random number generator

	

	int timeStepCounter =plasmaToQuantumTimestepRatio;  //how many quantum timesteps have we undergone?  every so often, do plasma stuff
  
  	// if it's a new run, initialize in normal way, otherwise read from annealed data files
  	if (newRun == 1)
  	{
  		init();
  	}
  	if (newRun == 0)
  	{
		readConditions(c0);
		recordedSpinUps=1;
	}
  	
  	while(t<=tmax+0.0009)                     // run simulation until tmax
	{
		
	  if(recordedSpinUps==0&&t>=tendV0){
	    measureSpinUps();
	    recordedSpinUps=1;
	    Zfunc(0);
	    printVAF(t);
	  }
	

		// other outputs like energy, vel_dist
	    if ((c0+1)%sampleFreq == 0&&timeStepCounter==1&&recordedSpinUps==1)
	    {
	    	output();
		Zfunc(1);
		printVAF(t);
	    }
	    if(timeStepCounter==plasmaToQuantumTimestepRatio){
	  	step();
	  	c0++;
		timeStepCounter=0;
	    }
	    if(t<tendV0 && t>tstartV0){
	      qstep();
	    }
	    else{
	      t+=quantumTimestep;
	    }
	    timeStepCounter++;
	}
  	//Print conditions
  	writeConditions(c0);
  	return 0;
}
