#include <armadillo>
#include "SIAM.h"
#include <chrono>

using iclock = std::chrono::high_resolution_clock;

int main() {

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

	arma::mat x_n = arma::zeros(nx, 2);

	std::chrono::duration<double> dur;
	auto start = iclock::now();
	for (arma::uword i = 0; i != xrange.n_elem; ++i) {
		double x = xrange(i);
		model.mfscf(x);
		x_n(i,0) = x;
		x_n(i,1) = model.n_imp;
	}

	dur = iclock::now() - start;
	std::cout << dur.count() << std::endl;
	x_n.save("SIAM_x_n.txt", arma::raw_ascii);


	return 0;
}
