#include "DecayRate.h"

DecayRate::DecayRate(double const& Gamma0_): 
	Gamma0(Gamma0_)
{}

double DecayRate::operator()(double const& x) {
	return Gamma0;
}
