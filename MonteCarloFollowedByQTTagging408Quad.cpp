//Monte Carlo + MD simulation of Yukawa potential for ions at arbitrary temperature + screening parameter
//Spits out VAF + g(r) + (others?  Somehow integrate with quantum code for simulating VCC?)

//NOTE: After MC simulation to equilibrate particle positions, run MD with 'collisions' for some amount of steps...
//by the nature of MC with finite particles, you won't necessarily get something that has exactly the right...
//correlations for a system of whatever temperature you assigned.  MD with 'collisions' basically puts the MD...
//system in contact with a thermal bath for some finite amount of steps 'numPreRecordMDSteps'.  In practice...
//this means that for every particle during this time for every timestep, you roll a dice to decide if a 'collision' happened...
//the likelihood of a collision depends on the size of your timestep TIMES your collision rate.  If a collision happens...
//whatever particle you are rolling for will have it's velocity reset to a velocity determined by the thermal distribution...
//for whatever Gamma you have chosen

//ANOTHER NOTE: Everything here (MC and MD) uses the 'minimum image' convention (MIC) and periodic boundary conditions (PBC)
//MIC: for a force (MD) or potential (MC) calc, find the 'minimum image' in any of the cubes 'surrounding' the 'real' cube
//This basically just means either add L, subtract L, or don't do anything to each dimension of particle (2): whichever puts...
//particle (2) closer to particle 1.  Ex: if positions are (L/8, 7L/8, L/4) and (7L/8, L/8, and L/2), to calculate forces,
//pretend as if particle (2) is actually at (-L/8, 9L/8, L/2) for purposes of calculating force.  In practice, you just do
//xDist -= L*round(xDist / L), etc.

//PBC: If during either MC or MD step a particle 'exits' the box (e.g., x,y,z <0 or >L), the particle is inserted back into
//the box 'pac-man' style (e.g., if particle moves to x = 9L/8 during a step, you reset its 'x' position to L/8.  If it...
//moves to Z=-L/4, move it to Z=3L/4, etc.).  This is handled by statements like:

//if (newx < 0) { newx += L; }
//if (newx > L) { newx -= L; }

//etc.

//comple command: g++ -std=c++11 -o tagQuad -O3 MCMDSpinTagQTSimQuad408.cpp -lm -I/home/tkl1/usr/include -L/users/tkl1/user/lib64


/*  We have also added the ability to simulate the optical pumping           */
/*  done by a cicrularly polarized on resonant 408 laser                     */
/*  After simulating the evolution of each ion's wavefunction                */
/*  for a user selected pump time, particles are tagged with                 */
/*  a probability <\uparrow|\psi> and the velocity of these                  */
/*  particles is recorded subsequently.  The <v^2(t)v^2(0)> is also recorded */
/*  starting at this time.  If the pumping is sufficiently quadratic,        */
/*  the two should match                                                     */

//include library headers + namespace

#define _USE_MATH_DEFINES
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<omp.h>
#include<sys/stat.h>
#include<iostream>
#include<random>
#include<string.h>
#include<sys/stat.h>
//#include<direct.h>
//#include<Windows.h>
//#include<ppl.h>
#include<omp.h>
#include<complex>
#include<armadillo>
//using namespace concurrency;
/*set up random number generator for initializing velocities + rollng die for MC + collisions in MD*/

using namespace std;
using namespace arma;
std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> uni(0, 1);
auto random_double = uni(rng);


/*input vars*/

/*save directory*/

char saveDirectory[256] = "dataSpinTagQuad/";//main directory.  A subfolder titled  Gamma_Kappa_Number___ will be created.  Subsubfolders for each job will be created.  data will be stored in the job folders;

//plasma variables (N, kappa, Gamma, etc.)

const int N = 4096;//number of particles in cubic lattice.  MAKE THIS A CUBE ROOT If for some reason you go over 21500 particles (you shouldnt...) then you'll need to boost "pairPairArraySize" above 450
const double kappa = 0.5;//screening parameter
const double Gamma = 3;//coupling constant, aka inverse 'normalized' temperature
const double n = 2;//density, in units 10^14 m^-3

//system variables (length, cut off radius, etc.)

double L = pow(N*4.*M_PI / 3., 1./3);//box length in units a  
double rCut = L / 2.;//cut off radius for forces and pair pair correlation calculation: needed for MC + MD calcs
double collisionFreq = 0.25;//odds of a collision during "1" time during MD equilibration post MC simulation (helps to equilibrate temperature should the MC not give you an exactly equilibrated environment (it won't!))
unsigned job;

//Monte carlo portion

const double monteCarloSteps = 100000;//how many steps of Monte Carlo you want to do to establish initial 'equilibrated' positions
const double maxRStep = 0.3;//max movement of a particle during a monte carlo step
double pairPairStep = 0.05;//step in units of 'a' for recording g(r)
double pairPairMax = L/2;//max value for recording g(r) (half the box length...g(r) meaningless beyond that point)
const int pairPairArraySize = 450;//Apparently have to do this manually?  Should just be pairPairMax/pairPairStep but for some reason it won't let me declare it as a constant if I type that Just went ahead and made it "big" (450)
//NOTE: everything above L/2 in a pairPair correlation is MEANINGLESS

//Molecular Dynamics Variables
std::normal_distribution<double> velocityDistribution(0, sqrt(1/Gamma));//set up velocity generator, spread given by sqrt(T_norm)...i.e. sqrt(1/Gamma)
const double timeStep = 0.005;//MD time step in 'normalized' time units (for us, it's einstein frequency, aka \omega_pi / sqrt(3)
const int numPreRecordMDSteps = 200;//number of MD steps for equilibration (e.g. w/ collisions) before recording (e.g. w/o collisions...collisions effect the parameters you're trying to record)
const int numVelAutoCorrsSteps = 1500;//number of COLLISIONLESS MD steps during which you record VAF, g(r), etc.


/*QUANTUM STUFF*/
//cooling parameters
std::complex<double> I(0,1);
double gamToEinsteinFreq = 174.07/sqrt(n);// ratio of gamma=1.41e8 to einstein freq
int plasmaToQuantumTimestepRatio = (int) round(87/sqrt(n));//half of above quantity because MD timestep is 0.005omega-1 while quantum timestep of 0.01gamma-1
double quantumTimestep = timeStep/plasmaToQuantumTimestepRatio;//timestep for quantum evolution (0.01Gam) in plasma time units
double plasVelToQuantVel=1.1821*pow(n,1./6);//conversion factor for going from plasma to quantum velocities (norm by a\omega and k/\gamma respectively)
double tpumpreal=0.0000001;// in seconds
double tpump = tpumpreal*813490*sqrt(n);//in omega_{E}t
int pumpMDTimeSteps = (int) round(tpump/timeStep);//how many MD time steps occur over the "pump" period
double detuning=0;//pump detuning in units \gamma_{408}
double Om=2;//rabi frequency in units \gamma_{408}
double decayRatio=0.0617;//ratio of D decay to S decay (8.7)/(141), see NIST spectral data
double vKick = 0.001208/plasVelToQuantVel;//vKick=\hbar*k/m in plasma units (it's 0.001208 in quantum units)
//waveFunctions
cx_mat wvFns[N+1000];
double numStates = 7;
mat ident = mat(numStates,numStates,fill::eye);
//states are same as in TKL doctoral thesis chapter 4
cx_mat wvFn1=cx_mat(ident.col(0),mat(numStates,1,fill::zeros));//S, mJ=-1/2
cx_mat wvFn2=cx_mat(ident.col(1),mat(numStates,1,fill::zeros));//S, mJ=+1/2
cx_mat wvFn3=cx_mat(ident.col(2),mat(numStates,1,fill::zeros));//P, mJ=+3/2
cx_mat wvFn4=cx_mat(ident.col(3),mat(numStates,1,fill::zeros));//P, mJ=+1/2
cx_mat wvFn5=cx_mat(ident.col(4),mat(numStates,1,fill::zeros));//P, mJ=-1/2
cx_mat wvFn6=cx_mat(ident.col(5),mat(numStates,1,fill::zeros));//P, mJ=-3/2
cx_mat wvFn7=cx_mat(ident.col(6),mat(numStates,1,fill::zeros));//All D states (no repump lasers so the substructure doesn't matter)

//decay coupling and rates
cx_mat cs[10];
double gs[10];

/*arrays*/
double R[3][N];// particle positions
double V[3][N];// particle velocities
double A[3][N];// particle accelerations
double PvelX[4001];          // velocity distribution
double PvelY[4001];          // velocity distribution
double PvelZ[4001];          // velocity distribution
double vel[4001];           // corresponding velocity bins
bool tagged[N];// list of particles tagged with probability |\langle \up | \psi \rangle |^{2}. where \psi is wavefunction after QT "pump" stage
double vStore[3][N][numVelAutoCorrsSteps]; //stored velocities for VAF
//double T[numVelAutoCorrsSteps+numAnisotropySteps+numReestablishEquilSteps];// Temperature: record during COLLISIONLESS MD to ensure that temperature is not changing dramatically.  Stop recording when 'heating/cooling' force is applied
double U[N];//contributions to potential energy from every particle: useful for MC calculations
double VAF[numVelAutoCorrsSteps];//At end of sim, this will store VAF calculated from vStore using 'recordVAF(void)'
double longViscAutoCorr[numVelAutoCorrsSteps];
double vCubeAutoCorr[numVelAutoCorrsSteps];
double vFourthAutoCorr[numVelAutoCorrsSteps];
double currentPotentialEnergy;//current potential energy evaluated during MC

/*Functions*/

void init(void);  //system initialization in cubic lattice
void MonteCarloStep(); //does Monte Carlo Step
void calculatePotentialEnergyForParticles();//Calculates U[i] after initialization: needed in order to perform subsequent MC steps
void changePotentialEnergy(int NPart, double randx, double randy, double randz);//changes U[] for given MC step.  If step accepted: new U[i] taken from this calculation
double calcUIJ(double totalDist);//U_ij
double calcAIJ(double totalDist);//A_IJ
void stepPositions();//step positions during MD step
void stepVelocities(double oldA[3][N]);//step velocities during MD step: requires 'old' accelerations (accelerations before stepping positions) and 'new' acceleration (accel after stepping position using 'old' velocities)...this is the Verlet algorithm
void recordTemperature(void);//write temperature to file
void calculateAccelerations(int tS);//calculate accelerations from forces on each particle based on input positions
void MDStep(int tS);//does molecular dynamics step (Verlet algorithm)
void recordPairPairCorr(int stepNum);//records the current g(r) (col 1: r/a, col 2: g(r/a)) and saves it in proper folder (...Gamma_Kappa_/pairPairCorrStepNum__.dat)
void recordVAF(void);//records the VAF from vStore (col 1: w_E * t, col 2: <v(w_E * t) dot v(0)>)
void recordLongViscAutoCorr(void);//records long visc autocorr
void recordVCubeAutoCorr(void);
void recordVFourthAutoCorr(void);

////////let's begin////////////

//define UIJ, AIJ (potential and acceleration).  Adaptable incase someone wants to use this for arbitrary U, F

double calcUIJ(double totalDist) {
	if (totalDist < rCut) {
		double UIJ = exp(-1 * kappa*totalDist) / totalDist;
		return UIJ;
	}
	else return 0;
}//UIJ

double calcAIJ(double totalDist) {
	if (totalDist < rCut) {
		double yukTerm = exp(-1 * kappa*totalDist);
		double AIJ = yukTerm*(pow(totalDist, -3) + kappa / (totalDist*totalDist));
		return AIJ;
	}
	else return 0;

}//AIJ

//initialization

void init() {
	//I decide to initialize with particles in cubic lattice.  Velocities are initialized from MB distribution given by 1/sqrt(Gamma) (i.e. sqrt(T_norm)).  wavefunctions are random combos of up and down w/ random phase

	int i, j, k, N0;
	N0 = 0;
	//cout << "aba\n";
	bool inSphere = false;
	//cout << L << "\n";
	//set particle positions (cubic lattice) + velocities (from MB distro)...positions start changing during MC, velocities begin changing in MD
	for (i = 0; i<round(pow(N, 1. / 3)); i++) {
		for (j = 0; j<round(pow(N, 1. / 3)); j++) {
			for (k = 0; k<round(pow(N, 1. / 3)); k++) {
				//R[0][N0] = uni(rng)*L;
				//R[1][N0] = uni(rng)*L;
				//R[2][N0] = uni(rng)*L;

				R[0][N0] = i*L/pow(N,1/3.) + 0.5;
				R[1][N0] = j*L / pow(N, 1 / 3.) + 0.5;
				R[2][N0] = k*L / pow(N, 1 / 3.) + 0.5;

				V[0][N0] = velocityDistribution(rng);
				V[1][N0] = velocityDistribution(rng);
				V[2][N0] = velocityDistribution(rng);
				//cout << R[0][N0] << "\n";
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
				wvFns[N0]=cx_mat(wvFn1Cont+wvFn2RealCont,wvFn2ImCont);
				
				
				N0++;

			}
		}
	}
	//cout << N0 << '\n';
	for(i=0;i<4001;i++)           // set up bins for velocity distribution
	  {                            // bin size is chosen as 0.0025 and range [-5:5]
	    vel[i]=(double)(i-2000)*0.0025;
	    
	  }
}

//Calculate initial potential energy for each particle U[i], needed for subsequent MC calcs

void calculatePotentialEnergyForParticles() {
	//sum up potential energy of 'nearest images'
	//double u=0;
	double currU = 0;//tabulated value of PE of particle I during sum over all other particles J
	double currx, curry, currz, compx, compy, compz, xDist, yDist, zDist, totalDistSq,totalDist;//temporary variables used in this function
	int i, j;//temp variables used in loops
	 
	for (i = 0; i < N; i++) {//for every particle I...
		compx = R[0][i];//positions of particle I
		compy = R[1][i];
		compz = R[2][i];
		
		for (j = 0; j < N; j++) {//sum over every particle J, adding PE_IJ to PE[I].
			//find nearest image
			currx = R[0][j];//positions of particle J
			curry = R[1][j];
			currz = R[2][j];
			xDist = compx - currx;//displacement between I + J along each dimension
			yDist = compy - curry;
			zDist = compz - currz;
			//find 'nearest' copy of particle (MIC)
			xDist -= L*round(xDist / L);
			yDist -= L*round(yDist / L);
			zDist -= L*round(zDist / L);

			//calc total distance + total distance squared
			totalDistSq = xDist*xDist + yDist*yDist + zDist*zDist;
			totalDist = sqrt(totalDistSq);

			
			if (j != i) {//ignore 'self' interaction
				currU += calcUIJ(totalDist);//add potential energy U_IJ to total U[i]
			}
		}//for every j
		U[i] = currU;//set U[i] to total summed energy
		currU = 0;//reset U tracker
	}//for every i

}

//calculate new U[i] for proposed MC Step

void changePotentialEnergy(int NPart, double randx, double randy, double randz) {
	//calculates energy difference for given MC step
	int j;
	double totalU, UijNew, UijOld, oldx, oldy, oldz, newx, newy, newz, currx, curry, currz, xDistNew, yDistNew, zDistNew, xDistOld, yDistOld, zDistOld, totalDistSqNew, totalDistSqOld,totalDistNew,totalDistOld;
	totalU = 0;
	UijNew = 0;
	UijOld = 0;
	//double rTruncSq = rTrunc*rTrunc;

	oldx = R[0][NPart];
	oldy = R[1][NPart];
	oldz = R[2][NPart];

	newx = R[0][NPart] + randx;
	newy = R[1][NPart] + randy;
	newz = R[2][NPart] + randz;
	//if particle left box during 'random' step, re-insert it in appropriate place
	if (newx < 0) { newx += L; }
	if (newx > L) { newx -= L; }
	if (newy < 0) { newy += L; }
	if (newy > L) { newy -= L; }
	if (newz < 0) { newz += L; }
	if (newz > L) { newz -= L; }

	for (j = 0; j < N; j++) {
		//find nearest image of unchanged particle
		currx = R[0][j];
		curry = R[1][j];
		currz = R[2][j];
		xDistNew = newx - currx;
		yDistNew = newy - curry;
		zDistNew = newz - currz;
		xDistOld = oldx - currx;
		yDistOld = oldy - curry;
		zDistOld = oldz - currz;
		//find 'nearest' copy of particle in periodic boundary conditions
		xDistNew -= L*round(xDistNew / L);
		yDistNew -= L*round(yDistNew / L);
		zDistNew -= L*round(zDistNew / L);
		xDistOld -= L*round(xDistOld / L);
		yDistOld -= L*round(yDistOld / L);
		zDistOld -= L*round(zDistOld / L);

		//calc total distance squared
		totalDistSqNew = xDistNew*xDistNew + yDistNew*yDistNew + zDistNew*zDistNew;
		totalDistSqOld = xDistOld*xDistOld + yDistOld*yDistOld + zDistOld*zDistOld;
		totalDistOld = sqrt(totalDistSqOld);
		totalDistNew = sqrt(totalDistSqNew);

		//change U[j] to reflect the change in U_IJ and tabulate new 'totalU' for U[i], which is changed at end of sum over J
		if (j != NPart) {
			UijNew = calcUIJ(totalDistNew);
			totalU += UijNew;
		}
		if (j != NPart) {
			UijOld = calcUIJ(totalDistOld);
			U[j] += (UijNew - UijOld);

		}
		UijNew = 0;//reset Uij
		UijOld = 0;
	}//for j

	U[NPart] = totalU;//replace old U[i] with new U[i] from sum over j of U_ij
}

void MonteCarloStep() {


	bool goodMove = false;
	bool inSphere = false;
	double randx, randy, randz, randPart, potentialEnergyDifference;
	int whichParticle;
	

	//roll for how far to move it within cube & see if it is within sphere (cube volume > sphere volume by factor about 0.5)
	while (inSphere == false) {//re-roll until you get move inside sphere...note this may not be technically correct to do (artificially inflates your # steps)
		randPart = uni(rng);
		whichParticle = (int)floor(randPart*N);
		randx = maxRStep*(2 * uni(rng) - 1);//pick dx,dy,dz between dr and -dr
		randy = maxRStep*(2 * uni(rng) - 1);
		randz = maxRStep*(2 * uni(rng) - 1);
		if (randx*randx + randy*randy + randz*randz < maxRStep*maxRStep) {
			inSphere = true;
		}//if in sphere
	}


	//calc potential energy U[i] after moving: if sum over U[i] lower, automatically accept move.  if not, roll for acceptance.  if not accepted, walk particle back
	double oldU[N];//record 'current' U[i] (before stepping)
	memcpy(oldU, U, sizeof(U));//copy over current U[i] vals incase we need to re-instate them
	changePotentialEnergy(whichParticle, randx, randy, randz);//calculate new U[i] after step
	potentialEnergyDifference = 0;
	for (int i = 0; i < N; i++) {//calculate potential energy difference between after and before step
		potentialEnergyDifference += U[i] - oldU[i];

	}
	
	if (potentialEnergyDifference < 0) {//if lower, accept move automatically
		goodMove = true;
		//cout << potentialEnergyDifference << "\n";
		//currentPotentialEnergy = potentialEnergy;
	}

	else {//if not lower, roll for acceptance w/ odds determined by boltzman factor exp(-\deltaU/2 *Gamma) (2 for double counting avoidance)
		double diceRoll = uni(rng);
		double acceptanceOdds = exp(-(potentialEnergyDifference / 2) *Gamma);
																			 
		if (diceRoll < acceptanceOdds) {
			goodMove = true;
		}

	}

	if (goodMove == true) {//if accepted
		//do move
		//change selected particles position
		R[0][whichParticle] += randx;
		R[1][whichParticle] += randy;
		R[2][whichParticle] += randz;
		//if particle left box, re-insert it in appropriate place
		if (R[0][whichParticle] < 0) { R[0][whichParticle] += L; }
		if (R[0][whichParticle] > L) { R[0][whichParticle] -= L; }
		if (R[1][whichParticle] < 0) { R[1][whichParticle] += L; }
		if (R[1][whichParticle] > L) { R[1][whichParticle] -= L; }
		if (R[2][whichParticle] < 0) { R[2][whichParticle] += L; }
		if (R[2][whichParticle] > L) { R[2][whichParticle] -= L; }
		//cout<<whichParticle<<"\n";
	}
	else {//if rejected
	    //re-instate old U vals (note: if accepted, 'changePotentialEnergy' already changed U[i]...so this is why we're changing U[i] back if move rejected!)
		memcpy(U, oldU, sizeof(U));
	}
}

//Now for MD Stuff!


void calculateAccelerations(int tS) {
  int i, j;
  double compx, compy, compz, currx, curry, currz, xDist, yDist, zDist, totalDistSq, totalDist, accel, dirForcePrefactor;//, fDotR;
  //fDotR = 0;
  //reset all accelerations
  for (i = 0; i < N; i++) {
    A[0][i] = 0;
    A[1][i] = 0;
    A[2][i] = 0;
  }
#pragma omp parallel private(i,j,compx,compy,compz,curry,currz,xDist,yDist,zDist,totalDistSq,totalDist,accel,dirForcePrefactor) shared(A,R,L,rCut)
  {
    //calculate new accelerations for this timestep
    //#pragma loop(hint_parallel(0))
#pragma omp for
    for (i = 0; i < N; i++) {//sum over i
      compx = R[0][i];
      compy = R[1][i];
      compz = R[2][i];
      //#pragma loop(hint_parallel(8)) 
      for (j = i + 1; j < N; j++) {//sum over j>i
	currx = R[0][j];
	curry = R[1][j];
	currz = R[2][j];
	xDist = compx - currx;
	yDist = compy - curry;
	zDist = compz - currz;
	//find 'nearest' copy of particle in periodic boundary conditions
	xDist -= L*round(xDist / L);
	yDist -= L*round(yDist / L);
	zDist -= L*round(zDist / L);

	//calc total distance squared
	totalDistSq = xDist*xDist + yDist*yDist + zDist*zDist;
	totalDist = sqrt(totalDistSq);

	//tally acceleration AIJ to AI and AJ
	if (totalDist < rCut) {
			  
	  dirForcePrefactor = exp(-1*kappa*totalDist)*(pow(totalDist, -3) + kappa / (totalDist*totalDist));
			  
	}
	else {
	  dirForcePrefactor=0;
	}
	dirForcePrefactor = calcAIJ(totalDist);
	accel = dirForcePrefactor*xDist;
	A[0][i] += accel;
	A[0][j] -= accel;
	accel = dirForcePrefactor*yDist;
	A[1][i] += accel;
	A[1][j] -= accel;
	accel = dirForcePrefactor*zDist;
	A[2][i] += accel;
	A[2][j] -= accel;



      }//sum over j
    }//sum over i
  }
}//calculate accelerations



void stepPositions() {//step positions according to verlet algorithm
	int i;
	for (i = 0; i < N; i++) {
		R[0][i] = R[0][i] + timeStep*V[0][i] + timeStep*timeStep / 2 * A[0][i];
		R[1][i] = R[1][i] + timeStep*V[1][i] + timeStep*timeStep / 2 * A[1][i];
		R[2][i] = R[2][i] + timeStep*V[2][i] + timeStep*timeStep / 2 * A[2][i];
		//if particle moved out of box, put it back in box and record what direction it moved in.
		if (R[0][i] < 0) { R[0][i] += L; }//BoxJumps[0][i] -= 1; }
		if (R[0][i] > L) { R[0][i] -= L; }//BoxJumps[0][i] += 1; }
		if (R[1][i] < 0) { R[1][i] += L; }//BoxJumps[1][i] -= 1; }
		if (R[1][i] > L) { R[1][i] -= L; }//BoxJumps[1][i] += 1; }
		if (R[2][i] < 0) { R[2][i] += L; } //BoxJumps[2][i] -= 1; }
		if (R[2][i] > L) { R[2][i] -= L; }//BoxJumps[2][i] += 1; }
	}

}

void stepVelocities(double oldA[3][N]) {//step velocities according to verlet algorithm.  ALSO handle collisions during 'collisional' MD for equilibration
  int i;
  double collRoll;
  
  
  for (i = 0; i < N; i++) {//for all i
    //roll for collision (NOTE: collisionFreq is set to zero in Main for 'collisionless' MD)
    collRoll = uni(rng);
    if (collRoll < timeStep*collisionFreq) {//collision probability equals timestep*collisionFreq
      //do collision
      V[0][i] = velocityDistribution(rng);
      V[1][i] = velocityDistribution(rng);
      V[2][i] = velocityDistribution(rng);
    }
    else {
      V[0][i] = V[0][i] + timeStep / 2 * (oldA[0][i] + A[0][i]);
      V[1][i] = V[1][i] + timeStep / 2 * (oldA[1][i] + A[1][i]);
      V[2][i] = V[2][i] + timeStep / 2 * (oldA[2][i] + A[2][i]);
    }

    
  }
  
}

void MDStep(int tS) {
	double oldA[3][N];
	memcpy(oldA, A, sizeof(A));//copy current acceleration values to be used in velocity verlet step
	stepPositions();//step positions using current V, A
	calculateAccelerations(tS);//calculate new acclerations after position step
	stepVelocities(oldA);//step velocities according to velocity-Verlet algorithm

}

//Quantum Trajectories

void qstep(void)
{  

  //422 pumping, no kick
  unsigned i;
  cx_mat wvFn;
  double velQuant;
  double velPlas;
  double tPart;
  cx_mat dpmatTerms[10];
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
    //tPart = t*gamToEinsteinFreq;
    for(int j=0;j<10;j++){
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
	cx_mat hamCouplingTermSP = -Om/2*wvFn2*wvFn6.t()*sqrt(gs[5])-Om/2*wvFn1*wvFn5.t()*sqrt(gs[2]);
	cx_mat hamEnergyTerm = totalDetRightSP*(wvFn3*wvFn3.t()+wvFn4*wvFn4.t())+totalDetLeftSP*(wvFn5*wvFn5.t()+wvFn6*wvFn6.t());
      
    
	cx_mat hamWithoutDecay=hamEnergyTerm+hamCouplingTermSP+hamCouplingTermSP.t();
	
	hamDecayTerm=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));
	for(int j=0;j<10;j++){
	  hamDecayTerm=hamDecayTerm-1./2*I*(gs[j]*cs[j].t()*cs[j]);
	}
	
	//cx_mat hamDecayTerm(zeros<mat>(numStates,numStates),-1*hamDecayTermComplex);
	
	cx_mat hamil=hamWithoutDecay+hamDecayTerm;
	
	//with hamiltonian calculated, can evolve wvFn using RK method

	double dtHalf = dtQuant*gamToEinsteinFreq/2;
	//get k1,y1 (k1 is slope at t0 calculated using y0, y0 is initial wvFn value.  y1 (wvFnk1) is wvFn stepped by dt/2 w/ slope k1)
	cx_mat dpmatTermsk1[10];
	cx_mat dpmatk1=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<10;j++){
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
	cx_mat dpmatTermsk2[10];
	cx_mat dpmatk2=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<10;j++){
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
	
	cx_mat dpmatTermsk3[10];
	cx_mat dpmatk3=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<10;j++){
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
	
	cx_mat dpmatTermsk4[10];
	cx_mat dpmatk4=cx_mat(zero_mat1,zero_mat1);
	for(int j=0;j<10;j++){
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
      double norm5 = std::norm(wvFn(4,0));
      double norm6 = std::norm(wvFn(5,0));
      double totalNorm = norm3+norm4+norm5+norm6;
      double prob3=norm3/totalNorm;
      double prob4=norm4/totalNorm;
      double prob5=norm5/totalNorm;
      double prob6=norm6/totalNorm;
      //wvFns[i]=wvFn;
      wvFn.zeros();
      //wvFn.print("wvFn:");
      //printf("\n%lg\n",prob4);
      double randDOrS = drand48();
      bool sDecay = true;
      double randDir = drand48();
      if(randDOrS<(decayRatio/(decayRatio+1))){
	sDecay=false;
      }
      
      if (rand2<prob3)
	{
	  if(sDecay){
	    wvFn(0,0).real(1);
	  }
	  else{
	      wvFn(6,0).real(1);
	    }
	  
	}
      else if(rand2<prob3+prob4)
	{
	  if(sDecay){
	    double rand3=drand48();
	    if(rand3<gs[1]){
	      wvFn(0,0).real(1);
	    }
	    else{
	      wvFn(1,0).real(1);
	    }
	  }
	  else{
	      wvFn(6,0).real(1);
	   
	    
	  }
	}
      else if(rand2<prob3+prob4+prob5)
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
	      wvFn(6,0).real(1);
	  }
	}
      else
	{
	  if(sDecay){
	    wvFn(1,0).real(1);
	  }
	  else{
	      wvFn(6,0).real(1);
	  }
	}
      //wvFns[i]=wvFn;
      //double randPhaseNum = drand48();
      //randPhase[i]=randPhaseNum*2*3.1419;//comment out for no random phase
      
      
 }
    
    wvFns[i]=wvFn;
  }//for all particles, evolve wave function
}


void recordVelsForAutocorrelations(int tS){
  int count;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < 3; j++) {
    
      vStore[j][i][tS] = V[j][i];
      
      count++;
    }
  }
}

void recordTemperature() {
	//write temperature to File

	FILE *fa;
	char buffer[256];
	char dataDirCopy[256];
	char fileName[256];
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,"temperature.dat"));
	fa=fopen(fileName, "a");
	double vSqMean = 0;
	int count = 0;
	for (int i = 0; i < N; i++) {
	  for (int j = 0; j < 3; j++) {
	    vSqMean += V[j][i] * V[j][i];
	    count++;
	  }
	}
	vSqMean = vSqMean / count;
	fprintf(fa, "%lg\n", vSqMean);
	fclose(fa);
}//recordTemperature



void recordPairPairCorr(int stepNum) {//calculate and record g(r)

	double pairPairCorr[pairPairArraySize] = { 0 };
	int i, j;

	double compx, compy, compz, currx, curry, currz, xDist, yDist, zDist, totalDistSq, totalDist;

	for (i = 0; i<N; i++) {//for every particle, calculate dist between nearest images of all other particles
		compx = R[0][i];
		compy = R[1][i];
		compz = R[2][i];

		for (j = 0; j<N; j++) {
			if (j != i)
			{
				//find nearest image
				currx = R[0][j];
				//cout<<currx<<"\n";
				curry = R[1][j];
				currz = R[2][j];
				xDist = compx - currx;
				yDist = compy - curry;
				zDist = compz - currz;
				//find 'nearest' copy of particle in periodic boundary conditions
				xDist -= L*round(xDist / L);
				yDist -= L*round(yDist / L);
				zDist -= L*round(zDist / L);

				//calc and bin total distance
				totalDistSq = xDist*xDist + yDist*yDist + zDist*zDist;
				totalDist = sqrt(totalDistSq);
				int bin = floor((int)(totalDist / pairPairStep));// r_ij/a floor'ed to nearest bin value
				if (bin<(int)(pairPairMax / pairPairStep)) {//if particles within minimum r/a value
					pairPairCorr[bin]++;//add 1 to bin
				}
			}//if not the same particle

		}//for all particles j
	}//for all particles i



	 //now normalize: need to divide all g(r) terms by N*4*\pi*r^2*dr*3/(4pi) (number in shell if particles randomly distributed)
	for (i = 0; i<(int)(pairPairMax / pairPairStep); i++) {
		if (i == 0) {//for bin 0, divide by sphere of N* 4/3 \pi dr^3
			pairPairCorr[i] = pairPairCorr[i] / (N * 4 / 3 * M_PI*pairPairStep*pairPairStep*pairPairStep);
		}
		else {//for all other bins, divide by N*4\pi*r^2*dr
			pairPairCorr[i] = pairPairCorr[i] / (N * 3 * pairPairStep*pairPairStep*pairPairStep*i*i);//i*pairPairStep=r
		}

	}//end normalization



	 //now write to file
	FILE *fa;
	char buffer[256];
	char dataDirCopy[256];
	char fileName[256];
	strcpy(dataDirCopy,saveDirectory);
	snprintf(buffer, 256,"pairPairCorrStepNum%d.dat", (int)(stepNum));
	strcpy(fileName,strcat(dataDirCopy,buffer));
	fa=fopen(fileName, "w");
	for (i = 0; i<(int)(pairPairMax / pairPairStep); i++) {
		fprintf(fa, "%lg\t%lg\n", i*pairPairStep, pairPairCorr[i]);
	}
	fclose(fa);
}//calculate pair pair corr


void recordVAF(void)
//calculate VAF from vStore and write to file
{
	
	unsigned i, j, tDiff;
	double VAFSumCurr=0;
	FILE *fa;
	char buffer[256];
	char dataDirCopy[256];
	char fileName[256];

	for (tDiff = 0; tDiff<numVelAutoCorrsSteps; tDiff++)
	{
		for (i = 0; i<N; i++)
		{
			for (j = 0; j<numVelAutoCorrsSteps - tDiff; j++)
			{//average over all vStore values that are tDiff apart from eachother in time
				VAFSumCurr += vStore[0][i][j] * vStore[0][i][j + tDiff] + vStore[1][i][j] * vStore[1][i][j + tDiff] + vStore[2][i][j] * vStore[2][i][j + tDiff];

			}
		}
		VAFSumCurr = VAFSumCurr / (N*(numVelAutoCorrsSteps - tDiff));//average!
		VAF[tDiff] = VAFSumCurr;//record VAF value for t=tDiff
		VAFSumCurr = 0;//reset VAF sum before calculating next VAF[t]
		
		
	}
	//now that VAF[t] is calculated, record it (column 1 = w_e t, column 2 = VAF[w_e t])
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,"VAF.dat"));
	fa=fopen(fileName, "w");
	for (tDiff = 0; tDiff < numVelAutoCorrsSteps; tDiff++)
	{
	  //cout << VAF[tDiff] << '\n';
		fprintf(fa, "%lg\t%lg\n", tDiff*timeStep, VAF[tDiff]);
	}
	fclose(fa);

}//recordVAF

void recordLongViscAutoCorr(void) {

	unsigned i, j, tDiff;
	double longViscSumCurr = 0;
	FILE *fa;
	char buffer[256];
	char dataDirCopy[256];
	char fileName[256];

	for (tDiff = 0; tDiff<numVelAutoCorrsSteps; tDiff++)
	{
		for (i = 0; i<N; i++)
		{
			for (j = 0; j<numVelAutoCorrsSteps - tDiff; j++)
			{//average over all vStore values that are tDiff apart from eachother in time
				longViscSumCurr += pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) + pow(vStore[1][i][j], 2) * pow(vStore[1][i][j + tDiff], 2)+ pow(vStore[2][i][j], 2) * pow(vStore[2][i][j + tDiff], 2) - 3/(Gamma*Gamma);

			}
		}
		longViscSumCurr = longViscSumCurr / (N*(numVelAutoCorrsSteps - tDiff));//average!
		longViscAutoCorr[tDiff] = longViscSumCurr;//record VAF value for t=tDiff
		longViscSumCurr = 0;//reset VAF sum before calculating next VAF[t]


	}
	//now that LongViscAutoCorr[t] is calculated, record it (column 1 = w_e t, column 2 = VAF[w_e t])
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,"longViscAutoCorr.dat"));
	fa=fopen(fileName, "w");
	for (tDiff = 0; tDiff < numVelAutoCorrsSteps; tDiff++)
	{
	  //cout << longViscAutoCorr[tDiff] << '\n';
		fprintf(fa, "%lg\t%lg\n", tDiff*timeStep, longViscAutoCorr[tDiff]);
	}
	fclose(fa);

}// recordLongViscAutoCorr

void recordVCubeAutoCorr(void) {
	
	unsigned i, j, tDiff;
	double vCubeSumCurr = 0;
	FILE *fa;
	char buffer[256];
	char dataDirCopy[256];
	char fileName[256];

	for (tDiff = 0; tDiff<numVelAutoCorrsSteps; tDiff++)
	{
		for (i = 0; i<N; i++)
		{
			for (j = 0; j<numVelAutoCorrsSteps - tDiff; j++)
			{//average over all vStore values that are tDiff apart from eachother in time
				vCubeSumCurr += pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) * vStore[0][i][j] * vStore[0][i][j+tDiff] + pow(vStore[1][i][j],2) * pow(vStore[1][i][j + tDiff],2) * vStore[1][i][j] * vStore[1][i][j+tDiff] + pow(vStore[2][i][j],2) * pow(vStore[2][i][j + tDiff],2) * vStore[2][i][j] * vStore[2][i][j+tDiff];

			}
		}
		vCubeSumCurr = vCubeSumCurr / (N*(numVelAutoCorrsSteps - tDiff));//average!
		vCubeAutoCorr[tDiff] = vCubeSumCurr;//record VAF value for t=tDiff
		vCubeSumCurr = 0;//reset VAF sum before calculating next VAF[t]
		//cout << 1 << '\n';

	}
	//now that VCubeAutoCorr[t] is calculated, record it (column 1 = w_e t, column 2 = VAF[w_e t])
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,"vCubeAutoCorr.dat"));
	fa=fopen(fileName, "w");
	for (tDiff = 0; tDiff < numVelAutoCorrsSteps; tDiff++)
	{
	  //cout << vCubeAutoCorr[tDiff] << '\n';
		fprintf(fa, "%lg\t%lg\n", tDiff*timeStep, vCubeAutoCorr[tDiff]);
	}
	fclose(fa);

}// recordVCubeAutoCorr

void recordVFourthAutoCorr(void) {

	unsigned i, j, tDiff;
	double vFourthSumCurr = 0;
	FILE *fa;
	char buffer[256];
	char dataDirCopy[256];
	char fileName[256];
	for (tDiff = 0; tDiff<numVelAutoCorrsSteps; tDiff++)
	{
		for (i = 0; i<N; i++)
		{
			for (j = 0; j<numVelAutoCorrsSteps - tDiff; j++)
			{//average over all vStore values that are tDiff apart from eachother in time
				vFourthSumCurr += pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) * pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) + pow(vStore[1][i][j], 2) * pow(vStore[1][i][j + tDiff], 2) * pow(vStore[1][i][j], 2) * pow(vStore[1][i][j + tDiff], 2) + pow(vStore[2][i][j], 2) * pow(vStore[2][i][j + tDiff], 2) * pow(vStore[2][i][j], 2) * pow(vStore[2][i][j + tDiff], 2);

			}
		}
		vFourthSumCurr = vFourthSumCurr / (N*(numVelAutoCorrsSteps - tDiff));//average!
		vFourthAutoCorr[tDiff] = vFourthSumCurr;//record VAF value for t=tDiff
		vFourthSumCurr = 0;//reset VAF sum before calculating next VAF[t]


	}
	//now that vFourthAutoCorr[t] is calculated, record it (column 1 = w_e t, column 2 = VAF[w_e t])
	strcpy(dataDirCopy,saveDirectory);
	strcpy(fileName,strcat(dataDirCopy,"vFourthAutoCorr.dat"));
	fa=fopen(fileName, "w");
	
	for (tDiff = 0; tDiff < numVelAutoCorrsSteps; tDiff++)
	{
	  //cout << vFourthAutoCorr[tDiff] << '\n';
		fprintf(fa, "%lg\t%lg\n", tDiff*timeStep, vFourthAutoCorr[tDiff]);
	}
	fclose(fa);

}// recordVFourthAutoCorr


void tagParticles() {
  
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
    double norm5=std::norm(wvFn(4,0));
    if(rand<norm1+norm3){
      tagged[i]=true;
      numSpinUp++;
    }
    else if(rand<norm1+norm3+norm4){
      double rand2=drand48();
      if(rand2<2./3){
	tagged[i]=true;
	numSpinUp++;
      }
      else{
	tagged[i]=false;
      }
    }
    else if(rand<norm1+norm3+norm4+norm5){
      double rand3=drand48();
      if(rand3<1./3){
	tagged[i]=true;
	numSpinUp++;
      }
      else{
	tagged[i]=false;
      }

    }
    else{
      tagged[i]=false;
    }

  }

  
}//tagParticles

void recordTaggedParticleMoments(int step)
{
  //start with particles tagged with prob prop to v
  double V2=1./(2.*0.002*0.002);
  unsigned i,j,numTagged=0;
  double firstMom=0,secondMom=0,thirdMom=0,fourthMom=0;
  double currVx;
  FILE *fa;
  char buffer[256];
  char dataDirCopy[256];
  char fileName[256];
  for(i=0;i<4001;i++)
    {
        PvelX[i]=0.0;
	PvelY[i]=0.0;
	PvelZ[i]=0.0;
    }
  
  for (i=0;i<N;i++){
    currVx = V[0][i];
    if(tagged[i])
      {
	firstMom+=currVx;
	secondMom+=currVx*currVx;
	thirdMom+=currVx*currVx*currVx;
	fourthMom+=currVx*currVx*currVx*currVx;
	numTagged+=1;
      }
    for(j=0;j<4001;j++)
      {
	if(tagged[i]==1){
	  PvelX[j]+=exp(-V2*(vel[j]-V[0][i])*(vel[j]-V[0][i]));
	  PvelY[j]+=exp(-V2*(vel[j]-V[1][i])*(vel[j]-V[1][i]));
	  PvelZ[j]+=exp(-V2*(vel[j]-V[2][i])*(vel[j]-V[2][i]));
	}
      }
  }
  firstMom/=numTagged;
  secondMom/=numTagged;
  thirdMom/=numTagged;
  fourthMom/=numTagged;
  
  //cout << numTaggedOne << '\n';
  //cout << numTaggedTwo << '\n';
  //cout << numTaggedThree << '\n';
  //cout << numTaggedFour << '\n';

  strcpy(dataDirCopy,saveDirectory);
  strcpy(fileName,strcat(dataDirCopy,"taggedMoments.dat"));
  fa=fopen(fileName, "a");
  fprintf(fa, "%lg\t%lg\t%lg\t%lg\t%lg\n", step*timeStep, firstMom, secondMom, thirdMom, fourthMom);
  fclose(fa);

 //normalize  distribuition
  
  for(i=0;i<4001;i++) // normalize Pvel
    {
      PvelX[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
      PvelY[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
      PvelZ[i]/=(6.0*sqrt(2*M_PI*0.002*0.002));
    }
  
  // output distribution
  
  sprintf(buffer,"vel_distX_timestep%06d.dat",step);
  strcpy(dataDirCopy,saveDirectory);
  strcpy(fileName,strcat(dataDirCopy,buffer));
  fa=fopen(fileName,"w");
  for(i=0;i<4001;i++)
    {
      fprintf(fa,"%lg\t%lg\n",vel[i],PvelX[i]);
    }
  fclose(fa);
}

int main(int argc, char**argv)
{//where the magic happens:  Set order of events, etc

  //setup directory structure
  
  job=(unsigned)atof(argv[1]);       // input job label
  //make new main directory
  mkdir(saveDirectory,ACCESSPERMS);
  //make new sub directory
  char namebuf[256];
  char namebuf2[256];
  char saveDirBackup[256];
  //strcpy(saveDirBackup,saveDirectory);
  sprintf(namebuf,"Gamma%dKappa%dNumIons%dPumpTime%dDet%dOm%dDensity%d",(unsigned)(Gamma*100),(unsigned)(kappa*100),(unsigned)(N),(unsigned)(1000000000.*tpumpreal),(unsigned)(100.*abs(detuning)),(unsigned)(100.*Om),(unsigned) (10.*n));
  strcat(saveDirectory,namebuf);
  //add date and timestamp
  //time_t rawtime;
  //struct tm* timeinfo;
  //char st [80];
  //time (&rawtime);
  //timeinfo = localtime(&rawtime);
  //strftime(st,80,"Date%m%d%y",timeinfo);
  //strcat(saveDirectory,st);
  mkdir(saveDirectory,ACCESSPERMS);
  //make directory for given job
  sprintf(namebuf2,"/job%d/",job);
  strcat(saveDirectory,namebuf2);
  mkdir(saveDirectory,ACCESSPERMS);
  //saveDirectory is now of form "OriginalSaveDirectory/Gamma%d...etc/job1/"

 //have to define all this here for some reason instead of the global varaible section...
  cs[0] = wvFn1*wvFn3.t();
  cs[1] = wvFn1*wvFn4.t();
  cs[2] = wvFn1*wvFn5.t();
  cs[3] = wvFn2*wvFn4.t();
  cs[4] = wvFn2*wvFn5.t();
  cs[5] = wvFn2*wvFn6.t();
  cs[6] = wvFn7*wvFn3.t();
  cs[7] = wvFn7*wvFn4.t();
  cs[8] = wvFn7*wvFn5.t();
  cs[9] = wvFn7*wvFn6.t();
  gs[0]=1;
  gs[1]=2./3;
  gs[2]=1./3;
  gs[3]=1./3;
  gs[4]=2./3;
  gs[5]=1;
  gs[6]=decayRatio;
  gs[7]=decayRatio;
  gs[8]=decayRatio;
  gs[9]=decayRatio;
 
  //step 1: initialize positions + velocities
  init();
  
  //step 2: calculate initial U[i]
  calculatePotentialEnergyForParticles();
  
  //Step 3: do montecarlo steps
  int k;
  for (k = 0; k<monteCarloSteps; k++) {
    
    //record g(r) every 10000 steps and output current step num to command terminal
    if (k % 10000 == 0) {
      recordPairPairCorr(k);
      cout << k << "\n";
    }
    
    MonteCarloStep();
  }//Monte Carlo
  
  //Step 4: Do 'collisional MD'
  for (k = 0; k < numPreRecordMDSteps; k++) {
    if (k % 100 == 0) {
      cout << k << "\n";
      
    }
    
    
    MDStep(k);
  }

    //Step 5: Tag particles by QT
    collisionFreq = 0;//change this so that VAF, etc. can be calculated correctly!
    int timeStepCounter = 0;
    cout<<"pumpMDTimeSteps="<<pumpMDTimeSteps<<"\n";
    cout<<"quantumStepsPerMD="<<plasmaToQuantumTimestepRatio<<"\n";
    for (k=0;k<pumpMDTimeSteps;k++){
      for (int l=0;l<plasmaToQuantumTimestepRatio;l++){
	qstep();
      }
      MDStep(k);
    }
    tagParticles();
  
  //Step 6: Do 'collisionless MD' & record various Vel Autocorrs & tagged particles.  Skip this if for some reason you don't want to record these things
  for (k = 0; k < numVelAutoCorrsSteps; k++) {
    recordTaggedParticleMoments(k);
    if (k % 100 == 0) {
      cout << k << "\n";
      recordPairPairCorr(k);
    }
    recordTemperature();
    MDStep(k);
    recordVelsForAutocorrelations(k);
  }
  
  //STEP 7: record VAF
  recordVAF();
  recordLongViscAutoCorr();
  recordVCubeAutoCorr();
  recordVFourthAutoCorr();

  
}//main
