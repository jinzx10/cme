#include "TwoPara.h"
#include "FSSH.h"
#include <chrono>

double const pi = acos(-1.0);
using iclock = std::chrono::high_resolution_clock;

int main() {
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
	double Gamma = 0.002;

	iclock::time_point start = iclock::now();
	std::chrono::duration<double> dur;

	TwoPara model(E_mpt, E_fil, Gamma, 0.0, bath_width);

	double dt = 1;
	double nt = 100;
	FSSH fssh(&model, mass, dt, nt, Gamma);

	double x0 = 2.0;
	double v0 = std::sqrt(2*0.001/mass); // barrier height ~ 0.002
	fssh.initialize(0, x0, v0, 1.0, 0.0);
	fssh.propagate();

	dur = iclock::now() - start;
	std::cout << "time elapsed = " << dur.count() << " seconds" << std::endl;

	return 0;
}
