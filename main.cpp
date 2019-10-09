#include "TwoPara.h"

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

	return 0;
}
