// Copyright NVIDIA Corporation 2002-2005
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once
/** @file */

#include <dp/math/Vecnt.h>

namespace dp
{
  namespace math
  {

    /*! \brief Sphere class.
     *  \remarks This is a sphere in n-space, determined by a center point and a radius. A sphere is
     *  often used as a bounding sphere around some data, so there is a variety of routines to
     *  calculate the bounding sphere. */
    template<unsigned int n, typename T> class Spherent
    {
      public:
        /*! \brief Default constructor.
         *  \remarks The default sphere is initialized to be invalid. A sphere is considered to be
         *  invalid, if the radius is negative. */
        Spherent();

        /*! \brief Constructor by center and radius.
         *  \param center The center of the sphere.
         *  \param radius The radius of the sphere. */
        Spherent( const Vecnt<n,T> &center, T radius );

      public:
        /*! \brief Get the center of the sphere.
         *  \return The center of the sphere.
         *  \sa setCenter, getRadius */
        const Vecnt<n,T> & getCenter() const;

        /*! \brief Get the radius of the sphere.
         *  \return The radius of the sphere.
         *  \remarks A sphere is considered to be invalid if the radius is negative.
         *  \sa setRadius, getCenter, invalidate */
        T getRadius() const;

        /*! \brief Invalidate the sphere by setting the radius to a negative value.
         *  \remarks A sphere is considered to be invalid if the radius is negative.
         *  \sa getRadius, setRadius */
        void invalidate();

        /*! \brief Function call operator.
         *  \param p A reference to a constant point.
         *  \return The signed distance of the point \a p from the sphere.
         *  \remarks The distance is negative when \a p is inside the sphere, and it is positive if \a
         *  p is outside the sphere. */
        T operator()( const Vecnt<n,T> &p ) const;

        /*! \brief Set the center of the sphere.
         *  \param center A reference to the constant center point.
         *  \sa getCenter, setRadius */
        void setCenter( const Vecnt<n,T> &center );

        /*! \brief Set the radius of the sphere.
         *  \param radius The radius of the sphere.
         *  \sa getRadius, setRadius, invalidate */
        void setRadius( T radius );

      private:
        Vecnt<n,T>  m_center;
        T           m_radius;
    };

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    /*! \brief Determine the bounding sphere of a number of points.
     *  \param points A pointer to the points to include.
     *  \param numberOfPoints The number of points used.
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T>
      Spherent<n,T> boundingSphere( const Vecnt<n,T> * points, unsigned int numberOfPoints );

    /*! \brief Determine the bounding sphere of a number of points.
     *  \param points A random access iterator to the points to use.
     *  \param numberOfPoints The number of points used.
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T, typename RandomAccessIterator >
      Spherent<n,T> boundingSphere( RandomAccessIterator points, unsigned int numberOfPoints );

  #if 0
    /*! \brief Determine the bounding sphere of a number of points.
     *  \param points A random access iterator to the points to use.
     *  \param indexSet The indexset describing how to access the random access iterator
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T, typename RandomAccessIterator, typename isetType >
      Spherent<n,T> boundingSphere( RandomAccessIterator points, const dp::sg::core::IndexSetLock &iset );
  #endif

    /*! \brief Determine the bounding sphere of a number of indexed points.
     *  \param points A pointer to the points to use.
     *  \param indices A pointer to the indices to use.
     *  \param numberOfIndices The number of indices used.
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T>
      Spherent<n,T> boundingSphere( const Vecnt<n,T> * points, const unsigned int * indices
                                  , unsigned int numberOfIndices );

    /*! \brief Determine the bounding sphere of a number of indexed points.
     *  \param points A random access iterator to the points to use.
     *  \param indices A pointer to the indices to use.
     *  \param numberOfIndices The number of indices used.
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T, typename RandomAccessIterator>
    Spherent<n,T> boundingSphere( RandomAccessIterator points, const unsigned int * indices
                                , unsigned int numberOfIndices );

    /*! \brief Determine the bounding sphere of a number of strip-indexed points.
     *  \param points A pointer to the points to use.
     *  \param strips A pointer to the strips of indices used.
     *  \param numberOfStrips The number of strips used.
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T>
      Spherent<n,T> boundingSphere( const Vecnt<n,T> * points
                                  , const std::vector<unsigned int> * strips
                                  , unsigned int numberOfStrips );

    /*! \brief Determine the bounding sphere of a number of strip-indexed points.
     *  \param points A random access iterator to the points to use.
     *  \param strips A pointer to the strips of indices used.
     *  \param numberOfStrips The number of strips used.
     *  \return A small sphere around the given points.
     *  \remarks The sphere is not necessarily the smallest possible bounding sphere. */
    template<unsigned int n, typename T, typename RandomAccessIterator>
    Spherent<n,T> boundingSphere( RandomAccessIterator points
                                , const std::vector<unsigned int> * strips
                                , unsigned int numberOfStrips );

    /*! \brief Determine the bounding sphere of a sphere and a point.
     *  \param s A reference to the constant sphere to use.
     *  \param p A reference to the constant point to use.
     *  \return The bounding sphere around \a s and \a p.
     *  \remarks If \a p is inside of \a s, a sphere of the same size and position as \a s is
     *  returned. Otherwise, the smallest sphere enclosing both \a s and \a p is returned. */
    template<unsigned int n, typename T>
      Spherent<n,T> boundingSphere( const Spherent<n,T> & s, const Vecnt<n,T> & p );

    /*! \brief Determine the bounding sphere around two spheres.
     *  \param s0 A reference to the constant first sphere to enclose.
     *  \param s1 A reference to the constant second sphere to enclose.
     *  \return A bounding sphere around \a s0 and \a s1.
     *  \remarks If the one sphere is completely contained in the other, a sphere equaling that other
     *  sphere is returned. Otherwise, the smallest sphere enclosing both \a s0 and \a s1 is returned. */
    template<unsigned int n, typename T>
      Spherent<n,T> boundingSphere( const Spherent<n,T> & s0, const Spherent<n,T> &s1 );

    template<unsigned int n, typename T>
      Spherent<n,T> boundingSphere( const Spherent<n,T> * spheres, unsigned int numberOfSpheres );

    /*! \brief Determine the bounding sphere of the intersection of two spheres.
     *  \param s0 A reference to the constant first sphere.
     *  \param s1 A reference to the constant second sphere.
     *  \return A sphere around the intersection of \a s0 and \a s1. */
    template<unsigned int n, typename T>
      Spherent<n,T> intersectingSphere( const Spherent<n,T> & s0, const Spherent<n,T> & s1 );

    /*! \brief Determine if a sphere is valid.
     *  \param s A reference to a constant sphere.
     *  \return \c true, if \a s is valid, otherwise \c false.
     *  \remarks A sphere is considered to be valid, if it's radius is not negative. */
    template<unsigned int n, typename T>
      bool isValid( const Spherent<n,T> & s );

    /*! \brief Determine if a sphere is positive.
     *  \param s A reference to a constant sphere.
     *  \return \c true, if \a s is positive, otherwise \c false.
     *  \remarks A sphere is considered to be positive, if it's radius is positive. */
    template<unsigned int n, typename T>
      bool isPositive( const Spherent<n,T> & s );

    /*! \brief Test for equality of two spheres.
     *  \param s0 A constant reference to the first sphere to test.
     *  \param s1 A constant reference to the second sphere to test.
     *  \return \c true, if the centers of the spheres are equal and the radii differ less than the
     *  type specific epsilon. Otherwise \c false.
     *  \sa operator!=() */
    template<unsigned int n, typename T>
      bool operator==( const Spherent<n,T> & s0, const Spherent<n,T> & s1 );

    /*! \brief Test for inequality of two spheres.
     *  \param s0 A constant reference to the first sphere to test.
     *  \param s1 A constant reference to the second sphere to test.
     *  \return \c true, if the centers of the spheres are different or the radii differ more than
     *  the type specific epsilon. Otherwise \c false.
     *  \sa operator==() */
    template<unsigned int n, typename T>
    bool operator!=( const Spherent<n,T> & s0, const Spherent<n,T> & s1 );

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Convenience type definitions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    typedef Spherent<3,float>   Sphere3f;
    typedef Spherent<3,double>  Sphere3d;

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlined member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    template<unsigned int n, typename T>
    inline Spherent<n,T>::Spherent()
      : m_center(0.0f,0.0f,0.0f)
      , m_radius(-1.0f)
    {
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T>::Spherent( const Vecnt<n,T> &center, T radius )
      : m_center(center)
      , m_radius(radius)
    {
    }

    template<unsigned int n, typename T>
    inline const Vecnt<n,T> & Spherent<n,T>::getCenter() const
    {
      return( m_center );
    }

    template<unsigned int n, typename T>
    inline T Spherent<n,T>::getRadius() const
    {
      return( m_radius );
    }

    template<unsigned int n, typename T>
    inline void Spherent<n,T>::invalidate()
    {
      m_radius = T(-1);
    }

    template<unsigned int n, typename T>
    inline T Spherent<n,T>::operator()( const Vecnt<n,T> &p ) const
    {
      return( distance( m_center, p ) - m_radius );
    }

    template<unsigned int n, typename T>
    inline void Spherent<n,T>::setCenter( const Vecnt<n,T> &center )
    {
      m_center = center;
    }

    template<unsigned int n, typename T>
    inline void Spherent<n,T>::setRadius( T radius )
    {
      m_radius = radius;
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T> boundingSphere( const dp::math::Vecnt<n,T> *points, unsigned int numberOfPoints )
    {
      DP_ASSERT( points );
      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[0];
      bbox[1] = points[0];
      for ( unsigned int i=1 ; i<numberOfPoints ; i++ )
      {
        for ( unsigned int j=0 ; j<n ; j++ )
        {
          if ( points[i][j] < bbox[0][j] )
          {
            bbox[0][j] = points[i][j];
          }
          else if ( bbox[1][j] < points[i][j] )
          {
            bbox[1][j] = points[i][j];
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the minimal radius needed from that center points
      T minRadius(0);
      for ( unsigned int i=0 ; i<numberOfPoints ; i++ )
      {
        T d = lengthSquared( points[i] - center );
        if ( minRadius < d )
        {
          minRadius = d;
        }
      }
      return( Spherent<n,T>( center, sqrt( minRadius ) ) );
    }

    template<unsigned int n, typename T, typename RandomAccessIterator >
    inline Spherent<n,T> boundingSphere( RandomAccessIterator points, unsigned int numberOfPoints )
    {
      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[0];
      bbox[1] = points[0];
      for ( unsigned int i=1 ; i<numberOfPoints ; i++ )
      {
        for ( unsigned int j=0 ; j<n ; j++ )
        {
          if ( points[i][j] < bbox[0][j] )
          {
            bbox[0][j] = points[i][j];
          }
          else if ( bbox[1][j] < points[i][j] )
          {
            bbox[1][j] = points[i][j];
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the minimal radius needed from that center points
      T minRadius(0);
      for ( unsigned int i=0 ; i<numberOfPoints ; i++ )
      {
        T d = lengthSquared( points[i] - center );
        if ( minRadius < d )
        {
          minRadius = d;
        }
      }
      return( Spherent<n,T>( center, sqrt( minRadius ) ) );
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T> boundingSphere( const Vecnt<n,T> * points, const unsigned int * indices, unsigned int numberOfIndices )
    {
      DP_ASSERT( points && indices );
      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[indices[0]];
      bbox[1] = points[indices[0]];
      for ( unsigned int i=1 ; i<numberOfIndices ; i++ )
      {
        unsigned int idx = indices[i];
        for ( unsigned int j=0 ; j<n ; j++ )
        {
          if ( points[idx][j] < bbox[0][j] )
          {
            bbox[0][j] = points[idx][j];
          }
          else if ( bbox[1][j] < points[idx][j] )
          {
            bbox[1][j] = points[idx][j];
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the maximal radius needed from that center point
      T maxRadius(0);
      for ( unsigned int i=0 ; i<numberOfIndices ; i++ )
      {
        T d = lengthSquared( points[indices[i]] - center );
        if ( maxRadius < d )
        {
          maxRadius = d;
        }
      }
      return( Spherent<n,T>( center, sqrt( maxRadius ) ) );
    }

    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Spherent<n,T> boundingSphere( RandomAccessIterator points, const unsigned int * indices, unsigned int numberOfIndices )
    {
      DP_ASSERT( indices );
      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[indices[0]];
      bbox[1] = points[indices[0]];
      for ( unsigned int i=1 ; i<numberOfIndices ; i++ )
      {
        unsigned int idx = indices[i];
        for ( unsigned int j=0 ; j<n ; j++ )
        {
          if ( points[idx][j] < bbox[0][j] )
          {
            bbox[0][j] = points[idx][j];
          }
          else if ( bbox[1][j] < points[idx][j] )
          {
            bbox[1][j] = points[idx][j];
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the maximal radius needed from that center point
      T maxRadius(0);
      for ( unsigned int i=0 ; i<numberOfIndices ; i++ )
      {
        T d = lengthSquared( points[indices[i]] - center );
        if ( maxRadius < d )
        {
          maxRadius = d;
        }
      }
      return( Spherent<n,T>( center, sqrt( maxRadius ) ) );
    }

  #if 0
    template<unsigned int n, typename T, typename RandomAccessIterator, typename isetType >
    inline Spherent<n,T> boundingSphere( RandomAccessIterator points, const dp::sg::core::IndexSet * iset )
    {
      dp::sg::core::Buffer::DataLock reader( iset->getBuffer() );
      const isetType * indices = reader.getPtr<isetType>();
      unsigned int prIdx = iset->getPrimitiveRestartIndex();

      DP_ASSERT(iset->getNumberOfIndices()>1); // requires at least two points

      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[indices[0]];
      bbox[1] = points[indices[0]];

      for ( unsigned int i=1 ; i<iset->getNumberOfIndices() ; i++ )
      {
        unsigned int idx = indices[i];
        if( idx != prIdx )
        {
          for ( unsigned int j=0 ; j<n ; j++ )
          {
            if ( points[idx][j] < bbox[0][j] )
            {
              bbox[0][j] = points[idx][j];
            }
            else if ( bbox[1][j] < points[idx][j] )
            {
              bbox[1][j] = points[idx][j];
            }
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the maximal radius needed from that center point
      T maxRadius(0);
      for ( unsigned int i=0 ; i<iset->getNumberOfIndices() ; i++ )
      {
        // skip primitive restart index here too
        if( indices[i] != prIdx )
        {
          T d = lengthSquared( points[indices[i]] - center );
          if ( maxRadius < d )
          {
            maxRadius = d;
          }
        }
      }
      return( Spherent<n,T>( center, sqrt( maxRadius ) ) );
    }
  #endif

    template<unsigned int n, typename T>
    inline Spherent<n,T> boundingSphere( const Vecnt<n,T> * points, const std::vector<unsigned int> * strips, unsigned int numberOfStrips )
    {
      DP_ASSERT( points && strips );
      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[strips[0][0]];
      bbox[1] = points[strips[0][0]];
      for ( unsigned int i=0 ; i<numberOfStrips ; i++ )
      {
        for ( size_t j=0 ; j<strips[i].size() ; j++ )
        {
          unsigned int idx = strips[i][j];
          for ( unsigned int k=0 ; k<n ; k++ )
          {
            if ( points[idx][k] < bbox[0][k] )
            {
              bbox[0][k] = points[idx][k];
            }
            else if ( bbox[1][k] < points[idx][k] )
            {
              bbox[1][k] = points[idx][k];
            }
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the minimal radius needed from that center points
      T minRadius = 0.0f;
      for ( unsigned int i=0 ; i<numberOfStrips ; i++ )
      {
        for ( size_t j=0 ; j<strips[i].size() ; j++ )
        {
          T d = lengthSquared( points[strips[i][j]] - center );
          if ( minRadius < d )
          {
            minRadius = d;
          }
        }
      }
      return( Spherent<n,T>( center, sqrt( minRadius ) ) );
    }

    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Spherent<n,T> boundingSphere( RandomAccessIterator points, const std::vector<unsigned int> * strips, unsigned int numberOfStrips )
    {
      DP_ASSERT( strips );
      //  determine the bounding box
      Vecnt<n,T> bbox[2];
      bbox[0] = points[strips[0][0]];
      bbox[1] = points[strips[0][0]];
      for ( unsigned int i=0 ; i<numberOfStrips ; i++ )
      {
        for ( size_t j=0 ; j<strips[i].size() ; j++ )
        {
          unsigned int idx = strips[i][j];
          for ( unsigned int k=0 ; k<n ; k++ )
          {
            if ( points[idx][k] < bbox[0][k] )
            {
              bbox[0][k] = points[idx][k];
            }
            else if ( bbox[1][k] < points[idx][k] )
            {
              bbox[1][k] = points[idx][k];
            }
          }
        }
      }

      //  take the center of the bounding box as the center of the bounding sphere
      Vecnt<n,T> center = T(0.5) * ( bbox[0] + bbox[1] );

      //  and determine the minimal radius needed from that center points
      T minRadius = 0.0f;
      for ( unsigned int i=0 ; i<numberOfStrips ; i++ )
      {
        for ( size_t j=0 ; j<strips[i].size() ; j++ )
        {
          T d = lengthSquared( points[strips[i][j]] - center );
          if ( minRadius < d )
          {
            minRadius = d;
          }
        }
      }
      return( Spherent<n,T>( center, sqrt( minRadius ) ) );
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T> boundingSphere( const Spherent<n,T> & s, const Vecnt<n,T> & p )
    {
      Vecnt<n,T> center(s.getCenter());
      if ( !isValid( s ) )
      {
        return( Spherent<n,T>( p, T(0) ) );
      }
      else
      {
        T radius(s.getRadius());
        T dist = distance( center, p );
        if ( radius < dist )
        {
          //  the point is outside the sphere
          //  the new center is a weighted sum of the old and the point
          center = ( ( dist + radius ) * center + ( dist - radius ) * p ) / ( 2 * dist );
          //  and the new radius is half the sum of the distance and the radius
          radius = T(0.5) * ( radius + dist );
        }
        return( Spherent<n,T>( center, radius ) );
      }
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T> boundingSphere( const Spherent<n,T> & s0, const Spherent<n,T> & s1 )
    {
      DP_STATIC_ASSERT( !std::numeric_limits<T>::is_integer );
      Spherent<n,T> s;
      if ( !isValid( s0 ) )
      {
        s = s1;
      }
      else if ( !isValid( s1 ) )
      {
        s = s0;
      }
      else
      {
        T dist = distance( s0.getCenter(), s1.getCenter() );
        if ( dist + s1.getRadius() <= s0.getRadius() )
        {
          //  s1 is completely inside of s0, so return s0
          s = s0;
        }
        else if ( dist + s0.getRadius() <= s1.getRadius() )
        {
          //  s0 is completely inside of s1, so return s1
          s = s1;
        }
        else
        {
          //  the spheres don't include each other
          T maxRadius = std::max( s0.getRadius(), s1.getRadius() );
          if ( dist < std::numeric_limits<T>::epsilon() )
          {
            //  the centers are too close together to calculate the weighted sum, so just take the
            //  center of them (knowing, that the radii also have to be close together; see above)
            s.setCenter( T(0.5) * ( s0.getCenter() + s1.getCenter() ) );
            s.setRadius( T(0.5) * dist + maxRadius );
          }
          else
          {
            // The new center is a weighted sum of the two given.
            // The weights are calculated as follows, with c0, c1 the centers and r0, r1 the radii of the spheres:
            //  Vecnt<n,T> diff = c1 - c0               // vector from c0 to c1
            //  T dist = length( diff );                // distance between the centers
            //  Vecnt<n,T> A = c0 - r0 * diff / dist;   // let A be one point on the combined sphere, along the axis c0-c1, r0 to the "left" of c0
            //  Vecnt<n,T> B = c1 + r0 * diff / dist;   // let B be one point on the combined sphere, along the axis c0-c1, r1 to the "right" of c1
            //  Vecnt<n,T> C = 0.5 * ( A + B );         // C is the center of between points A and B, that is the center of the combined sphere
            //               = 0.5 * ( c0 - r0 * diff / dist + c1 + r1 * diff / dist );
            //               = 0.5 * ( c0 - r0 * ( c1 - c0 ) / dist + c1 + r1 * ( c1 - c0 ) / dist );
            //               = 0.5 * ( ( 1 + ro/dist - r1/dist ) * c0 + ( 1 + r1/dist - r0/dist ) * c1 );
            //               = ( ( dist + r0 - r1 ) / ( 2 * dist ) ) * c0 + ( ( dist + r1 - r0 ) / ( 2 * dist ) ) * c1;
            //               = w0 * c0 + w1* c1;
            //  Pay special attention to the braces around the difference of the radii!
            //  Those are needed in case the distance and the radii have substantially different ranges.
            T w0 = ( dist + ( s0.getRadius() - s1.getRadius() ) ) / ( 2 * dist );
            T w1 = ( dist + ( s1.getRadius() - s0.getRadius() ) ) / ( 2 * dist );
            s.setCenter( w0 * s0.getCenter() + w1 * s1.getCenter() );
            //  and the new radius is half the sum of the distance and the radii
            s.setRadius( T(0.5) * ( s0.getRadius() + dist + s1.getRadius() ) );
          }
        }
      }
      return( s );
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T> boundingSphere( const Spherent<n,T> * spheres, unsigned int numberOfSpheres )
    {
      Boxnt<n,T> bbox;
      for ( unsigned int i=0 ; i<numberOfSpheres ; i++ )
      {
        if ( isValid( spheres[i] ) )
        {
          bbox = boundingBox( bbox, boundingBox( spheres[i] ) );
        }
      }
      Vecnt<n,T> center = bbox.getCenter();
      T radius(-1);
      for ( unsigned int i=0 ; i<numberOfSpheres ; i++ )
      {
        if ( isValid( spheres[i] ) )
        {
          T r = distance( center, spheres[i].getCenter() ) + spheres[i].getRadius();
          if ( radius < r )
          {
            radius = r;
          }
        }
      }
      return( Spherent<n,T>( center, radius ) );
    }

    template<unsigned int n, typename T>
    inline Spherent<n,T> intersectingSphere( const Spherent<n,T> & s0, const Spherent<n,T> & s1 )
    {
      Vecnt<n,T> v = s1.getCenter() - s0.getCenter();
      T d = v.normalize();
      Spherent<n,T> s;   //  the default constructor of a Spherent<n,T> creates an empty sphere
      if ( d < ( s0.getRadius() + s1.getRadius() ) )
      {
        //  the spheres do intersect
        T ds = d * d;
        T r0s = square( s0.getRadius() );
        T r1s = square( s1.getRadius() );
        if ( ( ds + r0s ) <= r1s )
        {
          //  s0 is inside of s1 => intersection is s0
          s = s0;
        }
        else if ( ( ds + r1s ) <= r0s )
        {
          //  s1 is inside of s0 => intersection is s1
          s = s1;
        }
        else
        {
          //  nontrivial intersection here
          T e = ( ds + r0s - r1s ) / ( 2 * d );
          s.setRadius( sqrt( r0s - e*e ) );
          s.setCenter( s0.getCenter() + e * v );
        }
      }
      return( s );
    }

    template<unsigned int n, typename T>
    inline bool isValid( const Spherent<n,T> & s )
    {
      return( 0 <= s.getRadius() );
    }

    template<unsigned int n, typename T>
    inline bool isPositive( const Spherent<n,T> & s )
    {
      return( 0 < s.getRadius() );
    }

    template<unsigned int n, typename T>
    inline bool operator==( const Spherent<n,T> & s0, const Spherent<n,T> & s1 )
    {
      return(   ( s0.getCenter() == s1.getCenter() )
            &&  abs( s0.getRadius() - s1.getRadius() ) < std::numeric_limits<T>::epsilon() );
    }

    template<unsigned int n, typename T>
    inline bool operator!=( const Spherent<n,T> & s0, const Spherent<n,T> & s1 )
    {
      return( ! ( s0 == s1 ) );
    }

  } // namespace math
} // namespace dp
