#include "TwoPara.h"
#include "FSSH.h"
#include <chrono>

double const pi = acos(-1.0);
using iclock = std::chrono::high_resolution_clock;

int main() {
	double x0_mpt = 2;
	double x0_fil = 2.3;
	double omega = 0.002;
	double mass = 14000;
	double dE_fil = 0.0005;

	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double bath_width = 0.05;
	double Gamma = 1.0e-4;
	double Ef = 0.0;
	double beta = 1.0 / 0.02;

	iclock::time_point start = iclock::now();
	std::chrono::duration<double> dur;

	TwoPara model(E_mpt, E_fil, Gamma, Ef, bath_width);
	DecayRate Gamma_(Gamma);

	double dt = 10;
	double nt = 100;
	FSSH fssh(&model, mass, dt, nt, Gamma_, beta);

	double sigma_x = std::sqrt(0.5/mass/omega);
	double sigma_v = std::sqrt(omega/mass/2.0);
	double x0 = x0_mpt + arma::randn()*sigma_x;
	double v0 = arma::randn() * sigma_v;
	fssh.initialize(0, x0, v0, 1.0, 0.0);
	fssh.propagate();
	//fssh.x_t.print();

	dur = iclock::now() - start;
	std::cout << "time elapsed = " << dur.count() << " seconds" << std::endl;

	return 0;
}
