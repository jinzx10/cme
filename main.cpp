#include "TwoPara.h"
#include "FSSH.h"
#include <mpi.h>
#include <sstream>
#include <string>

#ifdef TIMING
#include <chrono>
using iclock = std::chrono::high_resolution_clock;
#endif

double const pi = acos(-1.0);

int main() {

	int id, nprocs;
	int num_trajs = 960;
	std::string datadir = "/home/zuxin/job/cme/data/";

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int local_num_trajs = num_trajs / nprocs;

	/////////////////////////////////////////////////////////////////////////////
	//							Two-Parabola model
	/////////////////////////////////////////////////////////////////////////////
	double x0_mpt = 2;
	double x0_fil = 3;
	double omega_mpt = 0.04;
	double omega_fil = 0.04;
	double mass = 7;
	double dE_fil = 0.001;

	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega_mpt * omega_mpt * 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega_fil * omega_fil * 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double bath_width = 0.04;
	double bath_center = 0;
	arma::uword nbath = 600;
	arma::vec bath = arma::linspace(bath_center-bath_width, bath_center+bath_width, nbath);
	double dos = 1.0 / ( bath(1) - bath(0) );

	double Gamma = 0.002;
	double V = sqrt(Gamma/2/pi/dos);
	arma::vec cpl = arma::ones(nbath) * V;

	TwoPara model(E_mpt, E_fil, bath, cpl, nbath/2);


	/////////////////////////////////////////////////////////////////////////////
	//							FSSH dynamics
	/////////////////////////////////////////////////////////////////////////////

	double dt = 1;
	double nt = 1000;

	// store all data
	arma::mat x_t;
	arma::mat v_t;
	arma::umat state_t;

	// store local data
	arma::mat local_x_t = arma::zeros(nt, local_num_trajs);
	arma::mat local_v_t = arma::zeros(nt, local_num_trajs);
	arma::umat local_state_t = arma::zeros<arma::umat>(nt, local_num_trajs);

#ifdef TIMING
	iclock::time_point start;
	std::chrono::duration<double> dur;
#endif

	if (id == 0) {
		x_t.zeros(nt, num_trajs);
		v_t.zeros(nt, num_trajs);
		state_t.zeros(nt, num_trajs);
#ifdef TIMING
		start = iclock::now();
#endif
	}

	FSSH fssh(&model, mass, dt, nt, Gamma);

	// Wigner quasi-probability of the harmonic ground state:
	// exp(-m*omega*x^2/hbar) * exp(-p^2/m/omega/hbar)
	double sigma_x = std::sqrt(0.5/mass/omega_mpt);
	double sigma_v = std::sqrt(omega_mpt/mass/2.0);
	for (int i = 0; i != local_num_trajs; ++i) {
		double x0 = x0_mpt + arma::randn()*sigma_x;
		double v0 = std::sqrt(2*0.002/mass) + arma::randn()*sigma_v; // diabatic barrier height ~ 0.002
		bool state0 = false;
		fssh.initialize(state0, x0, v0, 1.0, 0.0);
		fssh.propagate();
		local_x_t.col(i) = fssh.x_t;
		local_v_t.col(i) = fssh.v_t;
		local_state_t.col(i) = fssh.state_t;
	}

	::MPI_Gather(local_x_t.memptr(), local_x_t.n_elem, MPI_DOUBLE, x_t.memptr(), local_x_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_v_t.memptr(), local_v_t.n_elem, MPI_DOUBLE, v_t.memptr(), local_v_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_state_t.memptr(), local_state_t.n_elem, MPI_DOUBLE, state_t.memptr(), local_state_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		x_t.save(datadir+"x_E0002.txt", arma::raw_ascii);
		v_t.save(datadir+"v_E0002.txt", arma::raw_ascii);
		state_t.save(datadir+"state_E0002.txt", arma::raw_ascii);
#ifdef TIMING
		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;
#endif
	}

	::MPI_Finalize();

	return 0;
}
