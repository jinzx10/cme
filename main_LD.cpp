#include <cstdlib>
#include <sstream>
#include <string>
#include <iostream>
#include <mpi.h>
#include "Langevin.h"
#include "TwoPara.h"

#ifdef TIMING
#include <chrono>
using iclock = std::chrono::high_resolution_clock;
#endif

double const pi = acos(-1.0);

int main() {

	int id, nprocs;
	int num_trajs = 1920;
	std::string datadir = "/home/zuxin/job/cme/data/20191121/Langevin2/";
	std::string cmd;
	const char* system_cmd = nullptr;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int local_num_trajs = num_trajs / nprocs;

	/////////////////////////////////////////////////////////////////////////////
	//							Two-Parabola model
	/////////////////////////////////////////////////////////////////////////////
	double x0_mpt = 2;
	double x0_fil = 2.3;
	double omega= 0.002;
	double mass = 14000;
	double dE_fil = 0.0005;

	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double bath_width = 0.05;
	double mu = 0.0;
	double Gamma = 1.0e-4;

	TwoPara model(E_mpt, E_fil, Gamma, mu, bath_width);


	/////////////////////////////////////////////////////////////////////////////
	//							FSSH dynamics
	/////////////////////////////////////////////////////////////////////////////

	double dt = 5;
	double nt = 100000;
	double beta = 1.0 / 0.02;

	// global data
	arma::mat x_t;
	arma::mat v_t;

	// local data
	arma::mat local_x_t = arma::zeros(nt, local_num_trajs);
	arma::mat local_v_t = arma::zeros(nt, local_num_trajs);

#ifdef TIMING
	iclock::time_point start;
	std::chrono::duration<double> dur;
#endif

	if (id == 0) {
		x_t.zeros(nt, num_trajs);
		v_t.zeros(nt, num_trajs);
#ifdef TIMING
		start = iclock::now();
#endif
	}

	double gamma = 1e-4;
	Langevin ld(&model, mass, dt, nt, beta, gamma);
	arma::arma_rng::set_seed_random();

	// Wigner quasi-probability of the harmonic ground state:
	// exp(-m*omega*x^2/hbar) * exp(-p^2/m/omega/hbar)
	double sigma_x = std::sqrt(0.5/mass/omega);
	double sigma_v = std::sqrt(omega/mass/2.0);
	for (int i = 0; i != local_num_trajs; ++i) {
		double x0 = x0_mpt + arma::randn()*sigma_x;
		double v0 = arma::randn()*sigma_v;
		ld.initialize(x0, v0);
		ld.propagate();
		local_x_t.col(i) = ld.x_t;
		local_v_t.col(i) = ld.v_t;
	}

	::MPI_Gather(local_x_t.memptr(), local_x_t.n_elem, MPI_DOUBLE, x_t.memptr(), local_x_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_v_t.memptr(), local_v_t.n_elem, MPI_DOUBLE, v_t.memptr(), local_v_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		cmd = "mkdir -p " + datadir;
		system_cmd = cmd.c_str();
		std::system(system_cmd);

		x_t.save(datadir+"x_t.txt", arma::raw_ascii);
		v_t.save(datadir+"v_t.txt", arma::raw_ascii);
		
#ifdef TIMING
		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;
#endif
	}

	::MPI_Finalize();

	return 0;
}

