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

//comple command: g++ -std=c++11 -o simCode -O3 MCMD.cpp.  Or just run in visual studio

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
//using namespace concurrency;
/*set up random number generator for initializing velocities + rollng die for MC + collisions in MD*/

using namespace std;
std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> uni(0, 1);
auto random_double = uni(rng);


/*input vars*/

/*save directory*/

char saveDirectory[256] = "data/";//main directory.  A subfolder titled  Gamma_Kappa_Number___ will be created.  Subsubfolders for each job will be created.  data will be stored in the job folders;

//plasma variables (N, kappa, Gamma, etc.)

const int N = 4096;//number of particles in cubic lattice.  MAKE THIS A CUBE ROOT
const double kappa = 0.5;//screening parameter
const double Gamma = 3;//coupling constant, aka inverse 'normalized' temperature
const double n = 0.4;//density, in units 10^14 m^-3

//system variables (length, cut off radius, etc.)

double L = pow(N*4.*M_PI / 3., 1./3);
double rCut = L / 2.;
double collisionFreq = 0.25;//odds of a collision during "1" time during MD equilibration post MC simulation (helps to equilibrate temperature should the MC not give you an exactly equilibrated environment (it won't!))
unsigned job;

//Monte carlo portion

const double monteCarloSteps = 200000;//how many steps of Monte Carlo you want to do to establish initial 'equilibrated' positions
const double maxRStep = 0.3;//max movement of a particle during a monte carlo step
double pairPairStep = 0.05;//step in units of 'a' for recording g(r)
double pairPairMax = L/2;//max value for recording g(r) (half the box length...g(r) meaningless beyond that point)
const int pairPairArraySize = 400;//Apparently have to do this manually?  Should just be pairPairMax/pairPairStep but for some reason it won't let me declare it as a constant if I type that

//Molecular Dynamics Variables
std::normal_distribution<double> velocityDistribution(0, sqrt(1/Gamma));//set up velocity generator, spread given by sqrt(T_norm)...i.e. sqrt(1/Gamma)
const double timeStep = 0.005;//MD time step in 'normalized' time units (for us, it's einstein frequency, aka \omega_pi / sqrt(3)
const int numPreRecordMDSteps = 200;//number of MD steps for equilibration (e.g. w/ collisions) before recording (e.g. w/o collisions...collisions effect the parameters you're trying to record)
const int numVelAutoCorrsSteps = 2500;//number of COLLISIONLESS MD steps during which you record VAF, g(r), etc.

//Temp Anisotropy Variables: Two stages: First stage the anisotropy is studied by instantaneously multiplying velocities (x by sqrt (1+tempPercentDiff), x and y by sqrt(1-tempPercentDiff/2)), as in Baalrud and Daligault's paper
//Then, following an equilibrating collisional stage ("reestablishEquilSteps"), the anisotropy is studied by heating or cooling (+ or - beta) the x axis for "anisotropyEstablishmentTime" (in us) then relaxes for anisotropyFromForcesRelaxSteps

//instantaneous anisotropy stuff
const int numInstantaneousAnisotropySteps = 2500;//number of collisionless md steps after you establish anisotropy by instantaneously changing Temps via tempPercentDiffs along different axes.
const int numReestablishEquilSteps = 500;//number of collisional md steps taken to reestablish equilibrium after the first round of anisotropy data is recorded
double tempPercentDiff = 0.15;//temperature differential to establish for "instantaneous" anisotropy data Tx->1.15Tx, Ty,Tz->0.925Ty,Tz for temp Perecent Diff of 0.15, for example


//'anisotropy establishment through anisotropic force application' stuff
int addLaserForce = 0;//will become 1 when force is applied then turned back to 0 after force concludes: see main();
bool applyForceAlongOneAxisOnly = false;//set this to true if you want to only heat (or cool, for negative beta) one axis.  False will apply 1/2 beta to x and -1/4 to y,z, analagous to heating/cooling along seperate axes in a way keeps total energy "constant" (it doesn't quite do that...but better than if you just pply force on one axis!)
const double beta = 26000;//cooling/heating (-/+) in units s^-1.  Measured 26000 for 20 MHz (see TKL thesis: note that this is half the value stated there since F=-beta/m*v for dT/dt = -2\beta*T), 
const int anisotropyEstablishmentTime = 10;//time for forces to establish anisotropy in us
int anisotropyEstablishingSteps = round(.8*anisotropyEstablishmentTime*sqrt(n) / timeStep);//number of steps in plasma units for which the anisotropic forces are applied
const int anisotropyFromForcesRelaxSteps = 2000;//number of collisionless md steps after you establish anisotropy by slowly changing Temps via anisotropic force application.

/*arrays*/
double R[3][N];// particle positions
double V[3][N];// particle velocities
double A[3][N];// particle accelerations
bool taggedOne[N];// list of particles tagged with probability .5+.5/3*(vx/vT).  If vx>3vT it's tagged automatically, if vx<3vT it's untagged automatically
bool taggedTwo[N];// list of particles tagged with probability .5+.5/9*(vx/vT)^2.  If |vx|>3vT it's tagged 50% of the time
bool taggedThree[N];// list of particles tagged with probability .5+.5/27*(vx/vT)^3.  If vx>3vT it's tagged automatically, if vx<3vT it's untagged automatically
bool taggedFour[N];// list of particles tagged with probability .5+.5/81*(vx/vT)^4.  If |vx|>3vT it's tagged 50% of the time
double meanVTagged[numVelAutoCorrsSteps];// mean velocity of tagged particles
double meanVSqTagged[numVelAutoCorrsSteps];// mean vSq tagged particles
double meanVCubeTagged[numVelAutoCorrsSteps];// mean v^3 of tagged particles
double meanVFourTagged[numVelAutoCorrsSteps];// mean v^4 of tagged particles
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
	//I decide to initialize with particles in cubic lattice.  Velocities are initialized from MB distribution given by 1/sqrt(Gamma) (i.e. sqrt(T_norm))

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
				N0++;

			}
		}
	}
	//cout << N0 << '\n';
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
    if (addLaserForce == 1)
      {
	if(applyForceAlongOneAxisOnly){
	  V[0][i] += V[0][i] * timeStep*1.234 * pow(10, -6) * beta / sqrt(n);//numerical factors here convert force to plasma units
	}
	else{
	  V[0][i] += V[0][i] * timeStep*1.234 * pow(10, -6) * beta / sqrt(n)/2;//numerical factors here convert force to plasma units
	  V[1][i] += V[1][i] * timeStep*1.234 * pow(10, -6) * beta / sqrt(n)/4*(-1);//numerical factors here convert force to plasma units
	  V[2][i] += V[2][i] * timeStep*1.234 * pow(10, -6) * beta / sqrt(n)/4*(-1);//numerical factors here convert force to plasma units
	}
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

void anisotropizeVelocities(){

  for (int i=0;i<N;i++){
    cout<<V[0][i]<<"\n";
    V[0][i]=sqrt(1+tempPercentDiff)*V[0][i];
    cout<<V[0][i]<<"\n";
    V[1][i]=sqrt(1-tempPercentDiff/2)*V[1][i];
    V[2][i]=sqrt(1-tempPercentDiff/2)*V[2][i];
  }

}//anisotropizeVelocities

void recordTempForEachAxis(char fileName[256],int step){
  double vSqMeanX = 0;
  double vSqMeanY = 0;
  double vSqMeanZ = 0;
  double vSqMean = 0;
  int count = 0;
  FILE *fa;
  for (int i = 0; i < N; i++) {
    vSqMeanX += V[0][i] * V[0][i];
    vSqMeanY += V[1][i] * V[1][i];
    vSqMeanZ += V[2][i] * V[2][i];
    count++;
  }

  vSqMeanX = vSqMeanX / count;
  vSqMeanY = vSqMeanY / count;
  vSqMeanZ = vSqMeanZ / count;
  fa = fopen(fileName,"a");
  fprintf(fa,"%lg\t%lg\t%lg\t%lg\n",step*timeStep,vSqMeanX,vSqMeanY,vSqMeanZ);
  fclose(fa);

}//record temperature along each axis


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
				longViscSumCurr += pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) + pow(vStore[1][i][j], 2) * pow(vStore[1][i][j + tDiff], 2)+ pow(vStore[2][i][j], 2) * pow(vStore[2][i][j + tDiff], 2) - 3/(Gamma*Gamma);//"extra" factor of 3 in subtracted term because we're adding values from all three axes

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
			  vFourthSumCurr += pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) * pow(vStore[0][i][j],2) * pow(vStore[0][i][j + tDiff],2) + pow(vStore[1][i][j], 2) * pow(vStore[1][i][j + tDiff], 2) * pow(vStore[1][i][j], 2) * pow(vStore[1][i][j + tDiff], 2) + pow(vStore[2][i][j], 2) * pow(vStore[2][i][j + tDiff], 2) * pow(vStore[2][i][j], 2) * pow(vStore[2][i][j + tDiff], 2)-3*9/(Gamma*Gamma*Gamma*Gamma);//"extra" factor of 3 in subtracted term because we're adding values from all three axes

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
  double roll,currVx;
  double vT = sqrt(1/Gamma);
  for (i=0;i<N;i++)
    {
      currVx = V[0][i];
      
      if(currVx<-3*vT)
	{
	  taggedOne[i]=false;
	}
      else if(currVx>3*vT)
	{
	  taggedOne[i]=true;
	}
      else
	{
	  roll = uni(rng);
	  if (roll<(.5+currVx/vT/6))
	    {
	      taggedOne[i]=true;
	    }
	  else
	    {
	      taggedOne[i]=false;
	    }
	}
	
      double c2 = .5/9/vT/vT;
      roll = uni(rng);
      if(currVx<-3*vT||currVx>3*vT)
	{
	  if (roll<.5)
	    {
	      taggedTwo[i]=false;
	    }
	  else
	    {
	      taggedTwo[i]=true;
	    }
	}
	 
      else
	{
	  if (roll<(c2*currVx*currVx))
	    {
	      taggedTwo[i]=true;
	    }
	  else
	    {
	      taggedTwo[i]=false;
	    }
	}
	

      
      double c3 = .5/27/vT/vT/vT;
      if(currVx<-3*vT)
	{
	  taggedThree[i]=false;
	}
      else if(currVx>3*vT)
	{
	  taggedThree[i]=true;
	}
      else
	{
	  roll = uni(rng);
	  if (roll<(.5+c3*currVx*currVx*currVx))
	    {
	      taggedThree[i]=true;
	    }
	  else
	    {
	      taggedThree[i]=false;
	    }
	}
	

    
      double c4 = .5/81/vT/vT/vT/vT;
      roll = uni(rng);
      if(currVx<-3*vT||currVx>3*vT)
	{
	  if (roll<.5)
	    {
	      taggedFour[i]=false;
	    }
	  else
	    {
	      taggedFour[i]=true;
	    }
	}
	 
      else
	{
	  if (roll<(c4*currVx*currVx*currVx*currVx))
	    {
	      taggedFour[i]=true;
	    }
	  else
	    {
	      taggedFour[i]=false;
	    }
	}
	
      
    }
  
}//tagParticles

void recordTaggedParticleMoments(int step)
{
  //start with particles tagged with prob prop to v

  unsigned i,numTaggedOne=0,numTaggedTwo=0,numTaggedThree=0,numTaggedFour=0;
  double firstMomOne=0,secondMomOne=0,thirdMomOne=0,fourthMomOne=0;
  double firstMomTwo=0,secondMomTwo=0,thirdMomTwo=0,fourthMomTwo=0;
  double firstMomThree=0,secondMomThree=0,thirdMomThree=0,fourthMomThree=0;
  double firstMomFour=0,secondMomFour=0,thirdMomFour=0,fourthMomFour=0;
  double currVx;
  FILE *fa;
  char buffer[256];
  char dataDirCopy[256];
  char fileName[256];
  for (i=0;i<N;i++){
    currVx = V[0][i];
    if(taggedOne[i])
      {
	firstMomOne+=currVx;
	secondMomOne+=currVx*currVx;
	thirdMomOne+=currVx*currVx*currVx;
	fourthMomOne+=currVx*currVx*currVx*currVx;
	numTaggedOne+=1;
      }
    if(taggedTwo[i])
      {
	firstMomTwo+=currVx;
	secondMomTwo+=currVx*currVx;
	thirdMomTwo+=currVx*currVx*currVx;
	fourthMomTwo+=currVx*currVx*currVx*currVx;
	numTaggedTwo+=1;
      }
    if(taggedThree[i])
      {
	firstMomThree+=currVx;
	secondMomThree+=currVx*currVx;
	thirdMomThree+=currVx*currVx*currVx;
	fourthMomThree+=currVx*currVx*currVx*currVx;
	numTaggedThree+=1;
      }
    if(taggedFour[i])
      {
	firstMomFour+=currVx;
	secondMomFour+=currVx*currVx;
	thirdMomFour+=currVx*currVx*currVx;
	fourthMomFour+=currVx*currVx*currVx*currVx;
	numTaggedFour+=1;
      }
  }
  firstMomOne/=numTaggedOne;
  secondMomOne/=numTaggedOne;
  secondMomOne-=1/(Gamma);//subtract long term equilibrium value
  thirdMomOne/=numTaggedOne;
  fourthMomOne/=numTaggedOne;
  fourthMomOne-=3/(Gamma*Gamma);//subract long term equilibrium value

  firstMomTwo/=numTaggedTwo;
  secondMomTwo/=numTaggedTwo;
  secondMomTwo-=1/(Gamma);
  thirdMomTwo/=numTaggedTwo;
  fourthMomTwo/=numTaggedTwo;
  fourthMomTwo-=3/(Gamma*Gamma);

  firstMomThree/=numTaggedThree;
  secondMomThree/=numTaggedThree;
  secondMomThree-=1/(Gamma);
  thirdMomThree/=numTaggedThree;
  fourthMomThree/=numTaggedThree;
  fourthMomThree-=3/(Gamma*Gamma);

  firstMomFour/=numTaggedFour;
  secondMomFour/=numTaggedFour;
  secondMomFour-=1/(Gamma);
  thirdMomFour/=numTaggedFour;
  fourthMomFour/=numTaggedFour;
  fourthMomFour-=3/(Gamma*Gamma);

  //cout << numTaggedOne << '\n';
  //cout << numTaggedTwo << '\n';
  //cout << numTaggedThree << '\n';
  //cout << numTaggedFour << '\n';

  strcpy(dataDirCopy,saveDirectory);
  strcpy(fileName,strcat(dataDirCopy,"taggedVOneMoments.dat"));
  fa=fopen(fileName, "a");
  fprintf(fa, "%lg\t%lg\t%lg\t%lg\t%lg\n", step*timeStep, firstMomOne, secondMomOne, thirdMomOne, fourthMomOne);
  fclose(fa);

  strcpy(dataDirCopy,saveDirectory);
  strcpy(fileName,strcat(dataDirCopy,"taggedVTwoMoments.dat"));
  fa=fopen(fileName, "a");
  fprintf(fa, "%lg\t%lg\t%lg\t%lg\t%lg\n", step*timeStep, firstMomTwo, secondMomTwo, thirdMomTwo, fourthMomTwo);
  fclose(fa);

  strcpy(dataDirCopy,saveDirectory);
  strcpy(fileName,strcat(dataDirCopy,"taggedVThreeMoments.dat"));
  fa=fopen(fileName, "a");
  fprintf(fa, "%lg\t%lg\t%lg\t%lg\t%lg\n", step*timeStep, firstMomThree, secondMomThree, thirdMomThree, fourthMomThree);
  fclose(fa);

  strcpy(dataDirCopy,saveDirectory);
  strcpy(fileName,strcat(dataDirCopy,"taggedVFourMoments.dat"));
  fa=fopen(fileName, "a");
  fprintf(fa, "%lg\t%lg\t%lg\t%lg\t%lg\n", step*timeStep, firstMomFour, secondMomFour, thirdMomFour, fourthMomFour);
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
  sprintf(namebuf,"Gamma%dKappa%dNumIons%d",(unsigned)(Gamma*100),(unsigned)(kappa*100),(unsigned)(N));
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
  
  
  //Step 5: Do 'collisionless MD' & record various Vel Autocorrs & tagged particles.  Skip this if for some reason you don't want to record these things
  collisionFreq = 0;//change this so that VAF, etc. can be calculated correctly!
  tagParticles();
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
  
  //STEP 6: record VAF
  recordVAF();
  recordLongViscAutoCorr();
  recordVCubeAutoCorr();
  recordVFourthAutoCorr();


  //STEP 7: Test temp anisotropy by instantaneously increasing x velocity by sqrt(1.15) while decreasing y and z velocities by sqrt(0.93)
  //Afterwards, do 'collisional' MD for a bit to reestablish equilibrium.
  anisotropizeVelocities();
  char fileName[256];
  strcpy(fileName,saveDirectory);
  strcat(fileName,"TemperaturesAlongAxesInstantaneous.dat");
  for (k=0;k<numInstantaneousAnisotropySteps;k++){
    recordTempForEachAxis(fileName,k);
    MDStep(k);

  }

  collisionFreq = 0.25;
  
  for (k = 0; k < numReestablishEquilSteps; k++) {
    if (k % 100 == 0) {
      cout << k << "\n";
      
    }
    
    
    MDStep(k);
  }
  
  //STEP 8: Test Temp Anisotropy by increasing or decreasing x velocity with a heating/cooling force, as we do in our experiment
  addLaserForce = 1;
  collisionFreq=0;
  //establish anisotropy
  strcpy(fileName,saveDirectory);
  strcat(fileName,"TemperaturesAlongAxesDuringForcePeriod.dat");
  for (k = 0; k < anisotropyEstablishingSteps; k++) {
    //char fileName[256];
    if (k % 100 == 0) {
      //cout << k << "\n";
      //recordPairPairCorr(k);
    }
    recordTempForEachAxis(fileName,k);
    MDStep(k);
  }

  // Let Anisotropy Relax
  //currentStage = 3;
  addLaserForce = 0;
  strcpy(fileName,saveDirectory);
  strcat(fileName,"TemperaturesAlongAxesAfterForcePeriod.dat");
  for (k = 0; k < anisotropyFromForcesRelaxSteps; k++) {
    if (k % 100 == 0) {
      //cout << k << "\n";
      //recordPairPairCorr(k);
    }
    recordTempForEachAxis(fileName,k);
    MDStep(k);
  }
  
}//main
