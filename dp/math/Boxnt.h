// Copyright NVIDIA Corporation 2002-2007
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
#include <limits>
#include <boost/assert.hpp>

namespace dp
{
  namespace math
  {

    /*! \brief Represents an axis-aligned box.
     *  \param n Specifies the box dimension.
     *  \param T Specifies the coordinate type used to represent the box.
     *  \remarks This class can be used to represent an axis-aligned minimum bounding box
     *  for a set of points in an \a n-dimensional space. 
     */
    template<unsigned int n, typename T>
    class Boxnt
    {
    public:
      /*! \brief Default-constructs an axis-aligned box.
       *  \remarks A default-constructed box initially is empty. */
      Boxnt(); 

      /*! \brief Copy-constructs an axis-aligned box. 
       *  \param rhs Specifies the master copy.
       *  \remarks After instantiation the box has the same extends as the master copy. */
      Boxnt(const Boxnt& rhs);

      /*! \brief Constructs an axis-aligned box out of two points.
       *  \param p0 Specifies the first point to consider.
       *  \param p1 Specifies the second point to consider. */
      Boxnt(const Vecnt<n,T>& p0, const Vecnt<n,T>& p1);

      /*! \brief Returns the lower edge of the box.
       *  \return The lower edge of the axis-aligned box. */
      const Vecnt<n,T>& getLower() const;

      /*! \brief Returns the upper edge of the box.
       *  \return The upper edge of the axis-aligned box. */
      const Vecnt<n,T>& getUpper() const;

      /*! \brief Returns the size of the box.
       *  \return The size of the axis-aligned box. */
      Vecnt<n,T> getSize() const;

      /*! \brief Returns the center of the box.
       *  \return The center of the axis-aligned box. */
      Vecnt<n,T> getCenter() const;

      /*! \brief Updates the box by a specified point.
       *  \param point The point by which to update the axis-aligned box.
       *  \remarks The function calculates new axis-aligned lower and upper 
       *  edges from the input point. */
      void update(const Vecnt<n,T>& point);

    private:
      
      void init(); // initially empties the box

      Vecnt<n,T>    m_lower; // lower edge of the box
      Vecnt<n,T>    m_upper; // upper edge of the box
    };

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    /*! \brief Instantiates an axis-aligned bounding box from a set of points.
     *  \param points Start address of input points.
     *  \param numberOfPoints Number of input points.
     *  \return Returns the bounding box calculated from the input points. */
    template<unsigned int n, typename T>
    Boxnt<n,T> boundingBox(const Vecnt<n,T> * points, unsigned int numberOfPoints);

    /*! \brief Instantiates an axis-aligned bounding box from a set of points.
     *  \param points Random access iterator of input points.
     *  \param numberOfPoints Number of input points.
     *  \return Returns the bounding box calculated from the input points. */
    template<unsigned int n, typename T, typename RandomAccessIterator>
    Boxnt<n,T> boundingBox(RandomAccessIterator points, unsigned int numberOfPoints);

    /*! \brief Instantiates an axis-aligned bounding box from a set of points.
     *  \param points Start address of input points.
     *  \param indices Start address of indices referencing the input points.
     *  \param numberOfIndices Number of indices.
     *  \return Returns the bounding box calculated from the input points 
     *  referenced by the specified indices. */
    template<unsigned int n, typename T>
    Boxnt<n,T> boundingBox(const Vecnt<n,T> * points, const unsigned int * indices, 
                           unsigned int numberOfIndices);

    /*! \brief Instantiates an axis-aligned bounding box from a set of points.
     *  \param points Random access iterator of input points.
     *  \param indices Start address of indices referencing the input points.
     *  \param numberOfIndices Number of indices.
     *  \return Returns the bounding box calculated from the input points 
     *  referenced by the specified indices. */
    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Boxnt<n,T> boundingBox(RandomAccessIterator points, const unsigned int * indices, 
                                  unsigned int numberOfIndices);

    /*! \brief Instantiates an axis-aligned bounding box from a set indexed triangle strips.
     *  \param points Start address of input points.
     *  \param indices Pointer to array of vectors of indices. Each vector contains one strip.
     *  \param numberOfStrips Number of strips.
     *  \return Returns the bounding box calculated from the input points 
     *  referenced by the specified indices. */
    template<unsigned int n, typename T>
    inline Boxnt<n,T> boundingBox(const Vecnt<n,T> * points, const std::vector<unsigned int> * strips, 
                                  unsigned int numberOfStrips);

    /*! \brief Instantiates an axis-aligned bounding box from a set indexed triangle strips.
     *  \param points Random access iterator of input points.
     *  \param indices Pointer to array of vectors of indices. Each vector contains one strip.
     *  \param numberOfStrips Number of strips.
     *  \return Returns the bounding box calculated from the input points 
     *  referenced by the specified indices. */
    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Boxnt<n,T> boundingBox(RandomAccessIterator points, const std::vector<unsigned int> * strips, 
                                  unsigned int numberOfStrips);

    /*! \brief Instantiates an axis-aligned bounding box as the union of two boxes.
     *  \param b0 Specifies the first bounding box.
     *  \param b1 Specifies the second bounding box.
     *  \return Returns the bounding box calculated as the union of \a b0 and \a b1. */
    template<unsigned int n, typename T>
    Boxnt<n,T> boundingBox(const Boxnt<n, T>& b0, const Boxnt<n, T>& b1);

    template<unsigned int n, typename T>
    Boxnt<n,T> boundingBox( const Spherent<n,T> & sphere );

    /*! \brief Determine if a box is valid.
     *  \param s b reference to a constant box.
     *  \return \c true, if \a b is valid, otherwise \c false.
     *  \remarks A box is considered to be valid, if none of its dimensions is negative. */
    template<unsigned int n, typename T>
      bool isValid( const Boxnt<n,T> & b );

    /*! \brief Determine if a box is positive.
     *  \param s A reference to a constant box.
     *  \return \c true, if \a b is positive, otherwise \c false.
     *  \remarks A box is considered to be positive, if all its dimensions are positive. */
    template<unsigned int n, typename T>
      bool isPositive( const Boxnt<n,T> & b );

    /*! \brief Compares two bounding boxes.
     *  \param lhs Specifies the left-hand bounding box.
     *  \param rhs Specifies the right-hand bounding box.
     *  \return The function returns \c true if both bounding boxes are equal. */
    template<unsigned int n, typename T>
    bool operator==(const Boxnt<n,T>& lhs, const Boxnt<n,T>& rhs);

    /*! \brief Compares two bounding boxes.
     *  \param lhs Specifies the left-hand bounding box.
     *  \param rhs Specifies the right-hand bounding box.
     *  \return The function returns \c true if both bounding boxes differ. */
    template<unsigned int n, typename T>
    bool operator!=(const Boxnt<n,T>& lhs, const Boxnt<n,T>& rhs);

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Convenience type definitions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    typedef Boxnt<2,float>      Box2f;
    typedef Boxnt<3,float>      Box3f;
    typedef Boxnt<4,float>      Box4f;
    typedef Boxnt<2,double>     Box2d;
    typedef Boxnt<3,double>     Box3d;
    typedef Boxnt<4,double>     Box4d;

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Implementation
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    template<unsigned int n, typename T>
    inline Boxnt<n,T>::Boxnt()
    {
      init();
    }

    template<unsigned int n, typename T>
    inline Boxnt<n,T>::Boxnt(const Boxnt<n,T>& rhs)
    : m_lower(rhs.m_lower)
    , m_upper(rhs.m_upper)
    {
    }

    template<unsigned int n, typename T>
    inline Boxnt<n,T>::Boxnt(const Vecnt<n,T>& p0, const Vecnt<n,T>& p1)
    {
      init(); // empty
      update(p0);
      update(p1);
    }

    template<unsigned int n, typename T>
    inline void Boxnt<n,T>::init()
    {
      for ( unsigned int i=0; i<n; ++i )
      {
        m_lower[i] = std::numeric_limits<T>::max();
        m_upper[i] = -std::numeric_limits<T>::max();
      }
    }

    template<unsigned int n, typename T>
    inline Vecnt<n,T> Boxnt<n,T>::getCenter() const
    {
      return( T(0.5) * ( m_lower + m_upper ) );
    }

    template<unsigned int n, typename T>
    inline Vecnt<n,T> Boxnt<n,T>::getSize() const
    {
      return( m_upper - m_lower );
    }

    template<unsigned int n, typename T>
    inline void Boxnt<n,T>::update(const Vecnt<n,T>& point)
    {
      for ( unsigned int i=0; i<n; ++i )
      {
        if ( m_lower[i] > point[i] ) 
        { 
          m_lower[i] = point[i]; 
        } 
        if ( m_upper[i] < point[i] ) 
        {
          m_upper[i] = point[i];
        }
      }
    }

    template<unsigned int n, typename T>
    inline const Vecnt<n,T>& Boxnt<n,T>::getLower() const
    {
      return m_lower;
    }

    template<unsigned int n, typename T>
    inline const Vecnt<n,T>& Boxnt<n,T>::getUpper() const
    {
      return m_upper;
    }

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlined non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    template<unsigned int n, typename T>
    inline Boxnt<n,T> boundingBox(const Vecnt<n,T> * points, unsigned int numberOfPoints)
    {
      DP_ASSERT(numberOfPoints>1); // requires at least two points
      Boxnt<n,T> bbox(points[0], points[0]); // initialize with first point
      for ( unsigned int i=1; i<numberOfPoints; ++i )
      {
        bbox.update(points[i]);
      }
      return bbox;
    }

    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Boxnt<n,T> boundingBox(RandomAccessIterator points, unsigned int numberOfPoints)
    {
      DP_ASSERT(numberOfPoints>1); // requires at least two points
      Boxnt<n,T> bbox(points[0], points[0]); // initialize with first point
      for ( unsigned int i=1; i<numberOfPoints; ++i )
      {
        bbox.update(points[i]);
      }
      return bbox;
    }

    template<unsigned int n, typename T>
    inline Boxnt<n,T> boundingBox(const Vecnt<n,T> * points, const unsigned int * indices, 
                                  unsigned int numberOfIndices)
    {
      DP_ASSERT(numberOfIndices>1); // requires at least two points
      Boxnt<n,T> bbox(points[indices[0]], points[indices[0]]); // initialize with first point
      for ( unsigned int i=1; i<numberOfIndices; ++i )
      {
        bbox.update(points[indices[i]]);
      }
      return bbox;
    }

    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Boxnt<n,T> boundingBox(RandomAccessIterator points, const unsigned int * indices, 
                                  unsigned int numberOfIndices)
    {
      DP_ASSERT(numberOfIndices>1); // requires at least two points
      Boxnt<n,T> bbox(points[indices[0]], points[indices[0]]); // initialize with first point
      for ( unsigned int i=1; i<numberOfIndices; ++i )
      {
        bbox.update(points[indices[i]]);
      }
      return bbox;
    }

    template<unsigned int n, typename T>
    inline Boxnt<n,T> boundingBox(const Vecnt<n,T> * points, const std::vector<unsigned int> * strips, 
                                  unsigned int numberOfStrips)
    {
      DP_ASSERT(numberOfStrips>0); // requires at least one strip
      DP_ASSERT(strips[0].size()>1); // requires at least two points
      Boxnt<n,T> bbox(points[strips[0][0]], points[strips[0][0]]); // initialize with first point
      for( unsigned int i=0; i<numberOfStrips; ++i )
      {
        // neglect re-processing of first point -> start with j=0
        for ( size_t j=0; j<strips[i].size(); ++j ) 
        {
          bbox.update(points[strips[i][j]]);
        }
      }
      return bbox;
    }

    template<unsigned int n, typename T, typename RandomAccessIterator>
    inline Boxnt<n,T> boundingBox(RandomAccessIterator points, const std::vector<unsigned int> * strips, 
                                  unsigned int numberOfStrips)
    {
      DP_ASSERT(numberOfStrips>0); // requires at least one strip
      DP_ASSERT(strips[0].size()>1); // requires at least two points
      Boxnt<n,T> bbox(points[strips[0][0]], points[strips[0][0]]); // initialize with first point
      for( unsigned int i=0; i<numberOfStrips; ++i )
      {
        // neglect re-processing of first point -> start with j=0
        for ( size_t j=0; j<strips[i].size(); ++j ) 
        {
          bbox.update(points[strips[i][j]]);
        }
      }
      return bbox;
    }

    template<unsigned int n, typename T>
    inline Boxnt<n,T> boundingBox(const Boxnt<n, T>& b0, const Boxnt<n, T>& b1)
    {
      if ( !isValid(b0) )
      {
        // take b1 if b0 is empty
        return Boxnt<n,T>(b1);
      }

      if ( !isValid(b1) )
      {
        // take b0 if b1 is empty
        return Boxnt<n,T>(b0);
      }

      // take union of both boxes
      Boxnt<n,T> bbox(b0);
      bbox.update(b1.getLower());
      bbox.update(b1.getUpper());
      return bbox;
    }

    template<unsigned int n, typename T>
    inline Boxnt<n,T> boundingBox( const Spherent<n,T> & sphere )
    {
      if ( !isValid( sphere ) )
      {
        return( Boxnt<n,T>() );
      }
      T radius = sphere.getRadius();
      Vecnt<n,T> offset;
      for ( unsigned int i=0 ; i<n ; i++ )
      {
        offset[i] = radius;
      }
      return( Boxnt<n,T>( sphere.getCenter() - offset, sphere.getCenter() + offset ) );
    }

    template<unsigned int n, typename T>
    inline bool isValid( const Boxnt<n,T> & b )
    {
      for ( unsigned int i=0; i<n; ++i ) 
      {
        if ( b.getUpper()[i] < b.getLower()[i] )
        {
          return false;
        }
      }
      return true;
    }

    template<unsigned int n, typename T>
    inline bool isPositive( const Boxnt<n,T> & b )
    {
      for ( unsigned int i=0; i<n; ++i ) 
      {
        if ( b.getUpper()[i] <= b.getLower()[i] )
        {
          return false;
        }
      }
      return true;
    }

    template<unsigned int n, typename T>
    inline bool operator==(const Boxnt<n,T>& lhs, const Boxnt<n,T>& rhs)
    {
      return lhs.getLower()==rhs.getLower() && lhs.getUpper()==rhs.getUpper();
    }

    template<unsigned int n, typename T>
    inline bool operator!=(const Boxnt<n,T>& lhs, const Boxnt<n,T>& rhs)
    {
      return !(lhs==rhs);
    }

  } // namespace math
} // namespace dp
