#pragma once

struct complex_t
{
	double x, y;
	_gt_hybrid complex_t(){}
	_gt_hybrid complex_t (const double& val) : x{val}, y{0.0} {}
	_gt_hybrid complex_t (const double& xx, const double& yy) : x{xx}, y{yy} {}
	
	_gt_hybrid complex_t operator ~ () const
	{
		return complex_t(x, -y);
	}
	
	_gt_hybrid complex_t operator + (const complex_t& rhs) const
	{
		return complex_t(x + rhs.x, y + rhs.y);
	}
	
	_gt_hybrid complex_t operator - (const complex_t& rhs) const
	{
		return complex_t(x - rhs.x, y - rhs.y);
	}
	
	_gt_hybrid complex_t operator * (const complex_t& rhs) const
	{
		return complex_t(x*rhs.x - y*rhs.y, x*rhs.y + y*rhs.x);
	}
	
	_gt_hybrid complex_t operator / (const complex_t& rhs) const
	{
		const auto denom = rhs.x*rhs.x + rhs.y*rhs.y;
		return complex_t ((x*rhs.x + y*rhs.y) / denom, (y*rhs.x - x*rhs.y) / denom);
	}
};

_gt_hybrid complex_t operator + (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) + rhs;
}

_gt_hybrid complex_t operator - (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) - rhs;
}

_gt_hybrid complex_t operator * (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) * rhs;
}

_gt_hybrid complex_t operator / (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) / rhs;
}