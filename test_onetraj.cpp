#include "TwoPara.h"
#include "FSSH.h"

double const pi = acos(-1.0);

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
	double bath_center = 0;
	arma::uword nbath = 800;
	arma::vec bath = arma::linspace(bath_center-bath_width, bath_center+bath_width, nbath);
	double dos = 1.0 / ( bath(1) - bath(0) );

	double Gamma = 0.002;
	double V = sqrt(Gamma/2/pi/dos);
	arma::vec cpl = arma::ones(nbath) * V;

	TwoPara model(E_mpt, E_fil, bath, cpl, nbath/2);

	double dt = 1;
	double nt = 100;
	FSSH fssh(&model, mass, dt, nt, Gamma);

	double x0 = 2.0;
	double v0 = std::sqrt(2*0.001/mass); // barrier height ~ 0.002
	fssh.initialize(0, x0, v0, 1.0, 0.0);
	fssh.propagate();

	return 0;
}
