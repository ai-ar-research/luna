/*********************                                                        */
/*! \file BoundedTensor.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BoundedTensor_h__
#define __BoundedTensor_h__

#include <iostream>

// avoid conflict with PyTorch's Warning macro
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>

// restore CVC4 Warning macro
#ifndef Warning
#define Warning (! ::CVC4::WarningChannel.isOn()) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

template <class T = torch::Tensor> class BoundedTensor
{
    typedef std::pair<T, T> Super;

public:
    BoundedTensor()
    {
    }

    BoundedTensor( const T &lower, const T &upper )
        : _container( lower, upper )
    {
    }

    // suppress implicit-copy deprecation
    BoundedTensor( const BoundedTensor<T> &other )
        : _container( other._container )
    {
    }

    T &lower()
    {
        return _container.first;
    }

    const T &lower() const
    {
        return _container.first;
    }

    T &upper()
    {
        return _container.second;
    }

    const T &upper() const
    {
        return _container.second;
    }

    BoundedTensor<T> &operator=( const BoundedTensor<T> &other )
    {
        _container = other._container;
        return *this;
    }

    T width() const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            return upper() - lower();
        } else {
            return upper() - lower();
        }
    }

    T center() const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            return (lower() + upper()) / 2.0;
        } else {
            return (lower() + upper()) / 2.0;
        }
    }

    bool operator==( const BoundedTensor<T> &other ) const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            return torch::all(torch::eq(lower(), other.lower())).template item<bool>() &&
                   torch::all(torch::eq(upper(), other.upper())).template item<bool>();
        } else {
            return _container == other._container;
        }
    }

    bool operator!=( const BoundedTensor<T> &other ) const
    {
        return !(*this == other);
    }

    bool operator<( const BoundedTensor<T> &other ) const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            T thisCenter = center();
            T otherCenter = other.center();
            T thisWidth = width();
            T otherWidth = other.width();

            if (torch::all(torch::eq(thisCenter, otherCenter)).template item<bool>()) {
                return torch::all(torch::lt(thisWidth, otherWidth)).template item<bool>();
            }
            return torch::all(torch::lt(thisCenter, otherCenter)).template item<bool>();
        } else {
            return _container < other._container;
        }
    }

protected:
    Super _container;
};

template <class T> std::ostream &operator<<( std::ostream &stream, const BoundedTensor<T> &boundedTensor )
{
    return stream << "[" << boundedTensor.lower() << "," << boundedTensor.upper() << "]";
}

#endif // __BoundedTensor_h__
