#ifndef __DECAY_RATE_H__
#define __DECAY_RATE_H__

struct DecayRate 
{
	DecayRate(double const& Gamma0_);

	double operator()(double const& x);
	double Gamma0;
};

#endif
