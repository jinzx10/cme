#include <armadillo>
#include "SIAM.h"
#include <chrono>
#include <mpi.h>

using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	double eps_d = -0.05;
	double g = 0.0075;
	double Gamma = 0.01;
	double W = 0.2;
	double U = 0.1;
	double pi = acos(-1.0);
	int n_bath = 300;
	int n_elec = 300;
	int n_val = 20;
	arma::vec bath = arma::linspace<arma::vec>(-W, W, n_bath);
	double dos = 1.0 / (bath(1) - bath(0));
	auto Ed = [&](double x) { return eps_d + sqrt(2) * g * x;};
	auto cpl = [&](double x) -> arma::vec {return arma::ones<arma::vec>(n_bath) * sqrt(Gamma/2/pi/dos);};

	SIAM model(Ed, cpl, U, bath, n_elec, n_val);

	double Ef_rough = bath(n_elec/2-1);
	double xmid = (Ef_rough - eps_d - U/2.0) / g;
	double xspan = (U/2 + Gamma*3) / g;
	int nx = 200;
	arma::vec xrange = arma::linspace<arma::vec>(xmid-xspan, xmid+xspan, nx);
	
	//nx = 1;
	//xrange = xrange(84);

	int nx_local = nx / nprocs;

	iclock::time_point start;
	std::chrono::duration<double> dur;
	arma::vec n_all;
	arma::mat data;

	if (id == 0) {
		start = iclock::now();
		n_all = arma::zeros(nx);
	}

	arma::vec n_local = arma::zeros(nx_local);
	for (int i = 0; i != nx_local; ++i) {
		double x = xrange(id*nx_local+i);
		model.mfscf(x);
		n_local(i) = x;
		n_local(i) = model.n_imp;
		std::cout << i+1+id*nx_local << "/" << nx << " finished" << std::endl;
	}

	::MPI_Gather(n_local.memptr(), nx_local, MPI_DOUBLE, n_all.memptr(), nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;

		data = arma::join_rows(xrange, n_all);
		data.save("data/SIAM_x_n.txt", arma::raw_ascii);
	}

	::MPI_Finalize();

	return 0;
}
