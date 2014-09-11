// Copyright NVIDIA Corporation 2009
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
#include <dp/util/StridedIterator.h>

namespace dp
{
  namespace math
  {
    /*! \brief Evaluate a bezier curve at parameter \a alpha.
     *  \param alpha The parameter to evaluate at. With \a alpha in the interval [0,1], you get a point on the curve
     *  between the end points.
     *  \param degree The degree of the bezier curve. For a cubic curve, it would be three.
     *  \param vertices A pointer to the \a degree + 1 control points of the bezier curve.
     *  \return The value of the bezier curve at parameter \a alpha.
     *  \sa frenetFrame, subdivide, BezierCurve */
    template<typename T> T    bezier( float alpha, unsigned int degree, const T * vertices );

    /*! \brief Evaluate a cubic bezier curve at parameter \a alpha.
     *  \param alpha The parameter to evaluate at. With \a alpha in the interval [0,1], you get a point on the curve
     *  between the end points.
     *  \param v0 The first of the four control points of a cubic bezier curve.
     *  \param v1 The second of the four control points of a cubic bezier curve.
     *  \param v2 The third of the four control points of a cubic bezier curve.
     *  \param v3 The fourth of the four control points of a cubic bezier curve.
     *  \return The value of the cubic bezier curve at parameter \a alpha. */
    template<typename T> T    bezier( float alpha, const T & v0, const T & v1, const T & v2, const T & v3 );

    //! \brief Cubic bezier interpolation with tangent.
    /*! \brief Evaluate a cubic bezier curve at parameter \a alpha, and also determine the tangent at that point.
     *  \param alpha The parameter to evaluate at. With \a alpha in the interval [0,1], you get a point on the curve
     *  between the end points.
     *  \param v0 The first of the four control points of a cubic bezier curve.
     *  \param v1 The second of the four control points of a cubic bezier curve.
     *  \param v2 The third of the four control points of a cubic bezier curve.
     *  \param v3 The fourth of the four control points of a cubic bezier curve.
     *  \param d A reference to the point to get the tangent at the evaluated point.
     *  \return The value of the cubic bezier curve at parameter \a alpha. */
    template<typename T> T    bezier( float alpha, const T & v0, const T & v1, const T & v2, const T & v3, T & d );

    /*! \brief Determine the frenet frame at a bezier curve at parameter \a alpha.
     *  \param alpha The parameter to evaluate at. With \a alpha in the interval [0,1], you get a frenet frame on the
     *  curve between the end points.
     *  \param degree The degree of the bezier curve. For a cubic curve, it would be three.
     *  \param vertices A pointer to the \a degree + 1 control points of the bezier curve.
     *  \param tangent A reference to the value to get the tangent at the specified point of the curve.
     *  \param normal A reference to the value to get the normal at the specified point of the curve.
     *  \param binormal A reference to the value to get the binormal at the specified point of the curve.
     *  \note The triple tangent/normal/binormal at a point of a curve is called Frenet frame. The tangent in the
     *  vector pointing along the first derivation of the curve. The binormal is orthogonal to the first and second
     *  derivative. That is, the normal is in the plane of the osculating circle to the curve at that point (the circle
     *  that has second-order contact with the curve at \a alpha).
     *  \sa bezier, subdivide, BezierCurve */
    template<typename T> void frenetFrame( float alpha, unsigned int degree, const T * vertices
                                         , T & tangent, T & normal, T & binormal );

    /*! \brief Subdivide a bezier curve at a parameter \a alpha.
     *  \param alpha The parameter to subdivide at. It has to be in the interval [0,1].
     *  \param degree The degree of the bezier curve. For a cubic curve, it would be three.
     *  \param vertices A pointer to the \a degree + 1 control points of the input bezier curve.
     *  \param leftVertices A pointer space to hold \a degree + 1 control points of the left sub-curve.
     *  \param rightVertices A pointer to space to hold \a degree + 1 control points of the right sub-curve.
     *  \sa bezier, subdivide, BezierCurve */
    template<typename T> void subdivide( float alpha, unsigned int degree, const T * vertices
                                       , T * leftVertices, T * rightVertices );

    /*! \brief A class to manage a bezier curve.
     *  \remarks A bezier curve of degree \a degree holds a control polyline of \a degree + 1 vertices.
     *  \sa BezierRect, BezierTriangle */
    template<typename T>
    class BezierCurve
    {
      public:
        /*! \brief Constructor out of an indexed set of vertices.
         *  \param degree The degree of the bezier curve. For a cubic curve, it would be three.
         *  \param vertices A pointer to the vertices for this curve.
         *  \note The vertices used for this BezierCurve are copied to local storage.
         *  \sa degree, evaluate, evaluateStrip */
        BezierCurve( unsigned int degree, const T * vertices );

        /*! \brief Constructor out of an indexed set of vertices.
         *  \param degree The degree of the bezier curve. For a cubic curve, it would be three.
         *  \param vertices A pointer to the vertices for this curve.
         *  \param indices An ConstIterator<unsigned int> that is a \a degree + 1 indices, into the array of \a vertices.
         *  \note The vertices used for this BezierCurve are copied to local storage.
         *  \sa degree, evaluate, evaluateStrip */
        BezierCurve( unsigned int degree, const T * vertices, const dp::sg::core::IndexSet::ConstIterator<unsigned int>&  indices );

        /*! \brief Get the degree of this BezierCurve.
         *  \return The degree of this BezierCurve.
         *  \sa evaluate, evaluateStrip */
        unsigned int  degree() const;

        /*! \brief Determine the derivative BezierCurve
         *  \return The derivative BezierCurve of this. */
        BezierCurve derivative() const;

        /*! \brief Evaluate the BezierCurve at the parameter \a t.
         *  \param t The parameter to evaluate the BezierCurve at. With \a t in the interval [0,1], you get a point
         *  on the curve between the end points.
         *  \return The point on the curve at parameter \a t.
         *  \sa degree, evaluateStrip */
        T             evaluate( float t );

        /*! \brief Evaluate the BezierCurve at \a count evenly spaced parameters.
         *  \param count The number of evaluations to do. It has to be at least 2.
         *  \param results A pointer to hold \a count evaluation results.
         *  \remark The BezierCurve is evaluate at the parameters 0, 1/(count-1), 2/(count-1),..., 1.
         *  \sa degree, evaluate */
        void          evaluateStrip( unsigned int count, T * results );

        /*! \brief Get the \a index's vertex of the control polygon
         *  \param index Index of the vertex to get.
         *  \remark The behaviour of this function is undefined, if \a index is equal to or larger than the number
         *  of control vertices in this BezierCurve. */
        const T &     operator[]( unsigned int index ) const;

      private:
        std::vector<T>  m_vertices;
    };

    /*! \brief Class to manage a rectangular bezier patch
     *  \remarks For a rectangular bezier patch of degrees \a degreeU and \a degreeV, there are
     *  (\a degreeU + 1 ) * (\a degreeV + 1 ) control points. 
     *  \a BezierCurve, BezierTriangle */
    template<typename T>
    class BezierRect
    {
      public:
        /*! \brief Constructor out of an indexed set of vertices.
         *  \param degreeU The degree of the patch in u-direction.
         *  \param degreeV The degree of the patch in v-direction.
         *  \param vertices A Buffer::ConstIterator of vertices for this patch.
         *  \param indices A pointer to ( \a degreeU + 1 ) * ( \a degreeV + 1 ) indices, into the array of \a vertices.
         *  \remarks The first \a degreeU + 1 points form the lower edge of the rectangle, while the last
         *  \a degree U + 1 points form the upper edge. With B00 specifying the lower left corner, Bu0 the lower right
         *  corner, B0v the upper left corner, and Buv the upper right corner, you've got the following correspondence
         *  to the array of vertices:
         *    (u==0, v==0)    B00 <-> vertices[indices[0]]
         *    (u==1, v==0)    Bu0 <-> vertices[indices[degreeU]]
         *    (u==0, v==1)    B0v <-> vertices[indices[(degreeU+1)*degreeV]]
         *    (u==1, v==1)    Buv <-> vertices[indices[(degreeU+1)*(degreeV+1)-1]] */
        BezierRect( unsigned int degreeU, unsigned int degreeV, typename dp::sg::core::Buffer::ConstIterator<T>::Type vertices, 
                    const dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices );

        /*! \brief Evaluate the BezierRect at the parameters \a u, \a v.
         *  \param t The parameter to evaluate the BezierCurve at. With \a t in the interval [0,1], you get a point
         *  on the curve between the end points.
         *  \return The point on the curve at parameter \a t.
         *  \sa evaluateMesh */
        T             evaluate( float u, float v );

        /*! \brief Evaluate the BezierRect at \a width * \a height evenly spaced parameters.
         *  \param width The number of evaluations to do in u-direction. It has to be at least 2.
         *  \param height The number of evaluations to do in v-direction. It has to be at least 2.
         *  \param results A pointer to hold \a width * \a height evaluation results.
         *  \remark The BezierRect is evaluate at the parameters
         *    (0,0),            (1/(width-1),0),            (2/(width-1),0),  ...,  (1,0),
         *    (0,1/(height-1)), (1/(width-1),1/(height-1)),                   ...,  (1,1/(height-1)),
         *    ....
         *    (0,1)             (1/(width-1),1),            (2/(width-1),1),  ...,  (1,1)
         *  \sa evaluate */
        void          evaluateMesh( unsigned int width, unsigned int height, T * results );

      private:
        unsigned int    m_degreeU;
        unsigned int    m_degreeV;
        std::vector<T>  m_vertices;
    };

    /*! \brief Class to manage a triangular bezier patch
     *  \remarks For a triangular bezier patch of degree \a degree, there are (\a degree + 1 ) * (\a degree + 2 ) / 2
     *  control points.
     *  \a BezierCurve, BezierRect */
    //  Bezier triangle with linear array of vertices. For degree 3, for general degree d,
    //  for example, the linear array of vertices has to be in the following manner:
    //  B300 <-> vertices[0]  (u==1)        Bd00 <-> vertices[0]                  (u==1)
    //  B030 <-> vertices[3]  (v==1)        B0d0 <-> vertices[d]                  (v==1)
    //  B003 <-> vertices[9]  (w==1)        B00d <-> vertices[((d+1)*(d+2))/2-1]  (w==1)
    template<typename T>
    class BezierTriangle
    {
      public:
        /*! \brief Constructor out of an indexed set of vertices.
         *  \param degree The degree of the patch.
         *  \param vertices An iterator of vertices for this patch.
         *  \param indices A pointer to ( \a degree + 1 ) * ( \a degree + 2 ) / 2 indices, into the array of \a vertices.
         *  \remarks The first \a degree + 1 points form the lower edge of the triangle, the next \a degree points form
         *  the row above the lower edge, while the last vertex is the top of the triangle. With Bd00 specifying the
         *  lower left corner (u==1), B0d0 the lower right corner (v==1), and B00d the top corner (w==1), you've got
         *  the following correspondence to the array of vertices:
         *    (u==1)    Bd00 <-> vertices[indices[0]]
         *    (v==1)    B0d0 <-> vertices[indices[degree]]
         *    (w==1)    B00d <-> vertices[indices[(degree+1)*(degree+2)/2-1]] */
        BezierTriangle( unsigned int degree, typename dp::sg::core::Buffer::ConstIterator<T>::Type vertices, 
                        const dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices );

        /*! \brief Evaluate the BezierTriangle at the barycentric coordinate ( \a u, \a v, 1 - \a u - \a v ).
         *  \param u The first baricentric coordinate to evaluate the BezierTriangle at.
         *  \param v The second baricentric coordinates to evaluate the BezierTriangle at.
         *  \return The point on the curve at the baricentric coordinate ( \a u, \a v, 1 - \a u - \a v ).
         *  \remarks You get an evaluation inside the triangular patch, if 0 <= \a u, 0 <= \a v, and \a u + \a v <= 1.
         *  \sa evaluateMesh */
        T     evaluate( float u, float v );

        /*! \brief Evaluate the BezierTriangle at \a size * ( \a size + 1 ) / 2 evenly spaced parameters.
         *  \param size The number of evaluations on an edge of the BezierTriangle.
         *  \param results A pointer to hold \a size * ( \a size + 1 ) / 2 evaluation results.
         *  \remark The BezierTriangle is evaluate at the parameters
         *    (1,0,0),          ((size-2)/(size-1),1/(size-1),0),                                   ...,  (0,1,0),
         *      ((size-2)/(size-1),0,1/(size-1)), ((size-3)/(size-1),1/(size-1),1/(size-1)),  ...,  (0,(size-2)/(size-1),1/(size-1)),
         *        ....
         *                                                  (0,0,1)
         *  \sa evaluate */
        void  evaluateMesh( unsigned int size, T * results );

      private:
        unsigned int    m_degree;
        std::vector<T>  m_vertices;
    };


    template<typename T>
    inline T bezier( float alpha, const T & v0, const T & v1, const T & v2, const T & v3 )
    {
      T b0 = lerp( alpha, v0, v1 );
      T b1 = lerp( alpha, v1, v2 );
      T b2 = lerp( alpha, v2, v3 );

      b0 = lerp( alpha, b0, b1 );
      b1 = lerp( alpha, b1, b2 );

      return( lerp( alpha, b0, b1 ) );
    }

    template<typename T>
    inline T bezier( float alpha, const T & v0, const T & v1, const T & v2, const T & v3, T & d )
    {
      T b0 = lerp( alpha, v0, v1 );
      T b1 = lerp( alpha, v1, v2 );
      T b2 = lerp( alpha, v2, v3 );

      b0 = lerp( alpha, b0, b1 );
      b1 = lerp( alpha, b1, b2 );

      d = b1 - b0;
      d.normalize();

      return( lerp( alpha, b0, b1 ) );
    }

    template<typename T>
    inline T bezier( float alpha, unsigned int degree, const T * vertices )
    {
      DP_ASSERT( 0 < degree );
      std::vector<T> tmp(degree);
      for ( unsigned int i=0 ; i<degree ; i++ )
      {
        tmp[i] = lerp( alpha, vertices[i], vertices[i+1] );
      }
      for ( unsigned int j=1 ; j<degree ; j++ )
      {
        for ( unsigned int i=0 ; i<degree-j ; i++ )
        {
          tmp[i] = lerp( alpha, tmp[i], tmp[i+1] );
        }
      }
      return( tmp[0] );
    }

    template<typename T> void frenetFrame( float alpha, unsigned int degree, const T * vertices
                                         , T & tangent, T & normal, T & binormal )
    {
      DP_ASSERT( 1 < degree );
      std::vector<T> tmp(vertices,vertices+degree+1);
      for ( unsigned int j=0 ; j<degree-2 ; j++ )
      {
        for ( unsigned int i=0 ; i<degree-j ; i++ )
        {
          tmp[i] = lerp( alpha, tmp[i], tmp[i+1] );
        }
      }
      dp::math::Vec3f secondDerivative = tmp[2] - 2 * tmp[1] + tmp[0];
      tangent = lerp( alpha, tmp[1], tmp[2] ) - lerp( alpha, tmp[0], tmp[1] );  // first derivative
      tangent.normalize();
      binormal = tangent ^ secondDerivative;
      binormal.normalize();
      normal = binormal ^ tangent;
    }

    template<typename T>
    inline void subdivide( float alpha, unsigned int degree, const T * vertices, T * leftVertices, T * rightVertices )
    {
      DP_ASSERT( false );   // never passed this path
      DP_ASSERT( ( T(0) <= alpha ) && ( alpha <= T(1) ) );
      DP_ASSERT( 0 < degree );

      leftVertices[0] = vertices[0];
      rightVertices[degree] = vertices[degree];
      for ( unsigned int i=0 ; i<degree ; i++ )
      {
        rightVertices[i] = lerp( alpha, vertices[i], vertices[i+1] );
      }
      for ( unsigned int j=1 ; j<degree ; j++ )
      {
        leftVertices[j] = rightVertices[0];
        for ( unsigned int i=0 ; i<degree-j ; i++ )
        {
          rightVertices[i] = lerp( alpha, rightVertices[i], rightVertices[i+1] );
        }
      }
    }


    template<typename T>
    inline BezierCurve<T>::BezierCurve( unsigned int degree, const T * vertices )
      : m_vertices(degree+1)
    {
      DP_ASSERT( vertices );
      for ( unsigned int i=0 ; i<=degree ; i++ )
      {
        m_vertices[i] = vertices[i];
      }
    }

    template<typename T>
    inline BezierCurve<T>::BezierCurve( unsigned int degree, const T * vertices, 
                                        const dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices )
      : m_vertices(degree+1)
    {
      DP_ASSERT( vertices );
      // MMM: leave this in??
      DP_ASSERT( false );   // never passed this path
      for ( unsigned int i=0 ; i<=degree ; i++ )
      {
        m_vertices[i] = vertices[indices[i]];
      }
    }

    template<typename T>
    inline unsigned int BezierCurve<T>::degree() const
    {
      return( dp::util::checked_cast<unsigned int>(m_vertices.size()-1) );
    }

    template<typename T>
    inline BezierCurve<T> BezierCurve<T>::derivative() const
    {
      unsigned int deg = degree();
      std::vector<T> degVertices( deg );
      for ( unsigned int i=0 ; i<deg ; i++ )
      {
        degVertices[i] = deg * ( m_vertices[i+1] - m_vertices[i] );
      }
      return( BezierCurve<T>( deg-1, &degVertices[0] ) );
    }

    template<typename T>
    inline T BezierCurve<T>::evaluate( float t )
    {
      DP_ASSERT( false );   // never passed this path
      return( bezier( t, degree(), &m_vertices[0] ) );
    }

    template<typename T>
    inline void BezierCurve<T>::evaluateStrip( unsigned int size, T * results )
    {
      DP_ASSERT( false );   // never passed this path
      DP_ASSERT( 1 < size );

      results[0] = m_vertices[0];
      results[size-1] = m_vertices[degree()];

      float stepSize = 1.0f / ( size - 1 );
      float t = stepSize;
      for ( unsigned int i=1 ; i<size-1 ; i++, t+=stepSize )
      {
        results[i] = bezier( t, degree(), &m_vertices[0] );
      }
    }

    template<typename T>
    inline const T & BezierCurve<T>::operator[]( unsigned int index ) const
    {
      DP_ASSERT( index < m_vertices.size() );
      return( m_vertices[index] );
    }


    template<typename VertexType, typename IndexType>
    inline BezierRect<T>::BezierRect( unsigned int degreeU, unsigned int degreeV, StridedConstIterator<T> vertices
                                    , IndexType const* indices, size_t numIndices )
      : m_degreeU(degreeU)
      , m_degreeV(degreeV)
      , m_vertices((degreeU+1)*(degreeV+1))
    {
      for ( size_t i = 0 ; i < numIndices; i++ )
      {
        m_vertices[i] = vertices[indices[i]];
      }
    }

    template<typename T>
    inline T BezierRect<T>::evaluate( float u, float v )
    {
      std::vector<T> tmp(m_degreeV+1);
      for ( unsigned int i=0, idx=0 ; i<=m_degreeV ; i++, idx+=m_degreeU+1 )
      {
        tmp[i] = bezier( u, m_degreeU, &m_vertices[idx] );
      }
      return( bezier( v, m_degreeV, &tmp[0] ) );
    }

    template<typename T>
    inline void BezierRect<T>::evaluateMesh( unsigned int width, unsigned int height, T * results )
    {
      DP_ASSERT( ( 1 < width ) && ( 1 < height ) );

      unsigned int meshNumberOfVertices = width * height;

      //  copy corner vertices to mesh
      results[0] = m_vertices[0];
      results[width-1] = m_vertices[m_degreeU];
      results[meshNumberOfVertices-width] = m_vertices[m_vertices.size()-1-m_degreeU];
      results[meshNumberOfVertices-1] = m_vertices[m_vertices.size()-1];

      float uStepSize = 1.0f / ( width - 1 );
      float vStepSize = 1.0f / ( height - 1 );

      //  calculate inner edge vertices, first on u-edges (v == 0.0 or v == 1.0)
      float u = uStepSize;
      for ( unsigned int i=1, j=meshNumberOfVertices-width+1 ; i<width-1 ; i++, j++, u+=uStepSize )
      {
        results[i] = bezier( u, m_degreeU, &m_vertices[0] );
        results[j] = bezier( u, m_degreeU, &m_vertices[m_vertices.size()-1-m_degreeU] );
      }
      // then on v-edges (u == 0.0 or u == 1.0) -> need to copy the edges in arrays
      std::vector<T> leftVertices( m_degreeV+1 );
      std::vector<T> rightVertices( m_degreeV+1 );
      for ( unsigned int i=0, j=0 ; i<=m_degreeV ; i++, j+=m_degreeU+1 )
      {
        leftVertices[i] = m_vertices[j];
        rightVertices[i] = m_vertices[j+m_degreeU];
      }
      float v = vStepSize;
      for ( unsigned int i=1, j=width ; i<height-1 ; i++, j+=width, v+=vStepSize )
      {
        results[j] = bezier( v, m_degreeV, &leftVertices[0] );
        results[j+width-1] = bezier( v, m_degreeV, &rightVertices[0] );
      }

      //  calculate inner vertices
      v = vStepSize;
      for ( unsigned int i=1, idx=width+1 ; i<width-1 ; i++, idx+=2, v+=vStepSize )
      {
        u = uStepSize;
        for ( unsigned int j=1 ; j<height-1 ; j++, idx++, u+=uStepSize )
        {
          results[idx] = evaluate( u, v );
        }
      }
    }

        template<typename VertexType, typename IndexType>
    inline BezierTriangle<T>::BezierTriangle( unsigned int degree, StridedConstIterator<T> vertices, 
                                              IndexType const* indices, size_t numIndices )
      : m_degree(degree)
      , m_vertices(((degree+1)*(degree+2))/2)
    {
      for ( size_t i = 0 ; i < numIndices; i++ )
      {
        m_vertices[i] = vertices[indices[i]];
      }
    }

    template<typename T>
    inline T BezierTriangle<T>::evaluate( float u, float v )
    {
      std::vector<T> b(m_vertices);

      float w = 1.0f - u - v;

      for ( unsigned int k=0 ; k<m_degree ; k++ )
      {
        unsigned int idx = 0;
        unsigned int off = 0;
        for ( unsigned int j=0 ; j<m_degree-k ; j++ )
        {
          for ( unsigned int i=0 ; i<m_degree-k-j ; i++ )
          {
            b[idx++] = u * b[off+i] + v * b[off+i+1] + w * b[off+i+1+m_degree-k-j];
          }
          off += m_degree - k - j + 1;
        }
      }

      return( b[0] );
    }

    template<typename T>
    inline void BezierTriangle<T>::evaluateMesh( unsigned int size, T * results )
    {
      DP_ASSERT( 1 < size );

      unsigned int meshNumberOfVertices = ( size * ( size + 1 ) ) / 2;

      //  copy corner vertices to mesh
      results[0] = m_vertices[0];
      results[size-1] = m_vertices[m_degree];
      results[meshNumberOfVertices-1] = m_vertices[m_vertices.size()-1];

      float stepSize = 1.0f / ( size - 1 );

      //  calculate inner edge vertices
      float u = ( size - 2 ) * stepSize;
      for ( unsigned int i=1, idx=size ; i<size-1 ; i++, u-=stepSize )
      {
        results[i]              = evaluate( u, 1 - u ); // bottom edge
        results[idx]            = evaluate( u, 0.0f );  // left edge
        results[idx+m_degree-i] = evaluate( 0.0f, u );  // right edge
        idx += size - i;
      }

      //  calculate inner vertices
      for ( unsigned int j=1, idx=size+1 ; j<size-2 ; j++, idx+=2 )
      {
        float u = ( size - 2 - j ) * stepSize;
        float v = stepSize;
        for ( unsigned int i=1 ; i<size-1-j ; i++, u-=stepSize, v+=stepSize, idx++ )
        {
          results[idx] = evaluate( u, v );
        }
      }
    }
  } // namespace math
} // namespace dp

