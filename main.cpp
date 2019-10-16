#include "TwoPara.h"
#include "FSSH.h"
#include <mpi.h>
#include <sstream>

double const pi = acos(-1.0);

int main() {

	int id, nprocs;
	int num_trajs = 100;

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
	arma::uword nbath = 800;
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
	double nt = 100;

	// store all data
	arma::mat x_t;
	arma::mat v_t;
	arma::umat state_t;

	// store local data
	arma::mat local_x_t = arma::zeros(nt, local_num_trajs);
	arma::mat local_v_t = arma::zeros(nt, local_num_trajs);
	arma::umat local_state_t = arma::zeros<arma::umat>(nt, local_num_trajs);

	if (id == 0) {
		x_t.zeros(nt, num_trajs);
		v_t.zeros(nt, num_trajs);
		state_t.zeros(nt, num_trajs);
	}

	FSSH fssh(&model, mass, dt, nt, Gamma);

	double sigma_x = std::sqrt(1.0/mass/omega_mpt);
	double sigma_v = std::sqrt(omega_mpt/mass);
	for (int i = 0; i != local_num_trajs; ++i) {
		double x0 = 2.0 + arma::randn()*sigma_x;
		double v0 = std::sqrt(2*0.001/mass) + arma::randn()*sigma_v; // diabatic barrier height ~ 0.002
		fssh.initialize(0, x0, v0, 1.0, 0.0);
		fssh.propagate();
		local_x_t.col(i) = fssh.x_t;
		local_v_t.col(i) = fssh.v_t;
		local_state_t.col(i) = fssh.state_t;
	}

	::MPI_Gather(local_x_t.memptr(), local_x_t.n_elem, MPI_DOUBLE, x_t.memptr(), local_x_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_v_t.memptr(), local_v_t.n_elem, MPI_DOUBLE, v_t.memptr(), local_v_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_state_t.memptr(), local_state_t.n_elem, MPI_DOUBLE, state_t.memptr(), local_state_t.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		x_t.save("x_t.txt", arma::raw_ascii);
		v_t.save("v_t.txt", arma::raw_ascii);
		state_t.save("state_t.txt", arma::raw_ascii);
	}


	::MPI_Finalize();

	return 0;
}
