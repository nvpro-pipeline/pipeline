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


#include <dp/sg/core/GeoNode.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>

using namespace dp::util;
using namespace dp::sg::core;

using std::map;
using std::vector;

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      DestrippingTraverser::DestrippingTraverser(void)
      {
      }

      DestrippingTraverser::~DestrippingTraverser(void)
      {
      }

      PrimitiveType convertLineStrip( Primitive * p, vector<unsigned int> & newIndices )
      {
        unsigned int count = p->getElementCount();
        DP_ASSERT( 1 < count );
        newIndices.reserve( 2 * ( count - 1 ) );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( oldIndices[0] != pri );
          for ( unsigned int i=1 ; i<count ; i++ )
          {
            if ( oldIndices[i] == pri )
            {
              i++;    // advance to the first vertex of the next line strip
              DP_ASSERT( ( count <= i ) || ( oldIndices[i] != pri ) );
            }
            else
            {
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[i] );
            }
          }
        }
        else
        {
          for ( unsigned int i=1 ; i<count ; i++ )
          {
            newIndices.push_back( i-1 );
            newIndices.push_back( i );
          }
        }
        return( PrimitiveType::LINES );
      }

      PrimitiveType convertLineLoop( Primitive * p, vector<unsigned int> & newIndices )
      {
        unsigned int count = p->getElementCount();
        DP_ASSERT( 1 < count );
        newIndices.reserve( 2 * count );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( oldIndices[0] != pri );
          unsigned int startI = 0;
          for ( unsigned int i=1 ; i<count ; i++ )
          {
            if ( oldIndices[i] == pri )
            {
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[startI] );
              i++;    // advance to the first vertex of the next line loop
              startI = i;
              DP_ASSERT( ( count <= i ) || ( oldIndices[i] != pri ) );
            }
            else
            {
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[i] );
            }
          }
          // if the very last index is _not_ pri, close the loop
          // the other case has been handled in the loop above
          if ( oldIndices[count-1] != pri )
          {
            newIndices.push_back( oldIndices[count-1] );
            newIndices.push_back( oldIndices[startI] );
          }
        }
        else
        {
          newIndices.push_back( 0 );
          for ( unsigned int i=1 ; i<count ; i++ )
          {
            newIndices.push_back( i );
            newIndices.push_back( i );
          }
          newIndices.push_back( 0 );
        }
        return( PrimitiveType::LINES );
      }

      PrimitiveType convertTriangleStrip( Primitive * p, vector<unsigned int> & newIndices )
      {
        unsigned int count = p->getElementCount();
        DP_ASSERT( 2 < count );
        newIndices.reserve( 3 * ( count - 2 ) );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( ( oldIndices[0] != pri ) && ( oldIndices[1] != pri ) );
          for ( unsigned int i=2, odd=1 ; i<count ; i++, odd ^= 1 )
          {
            if ( oldIndices[i] == pri )
            {
              i += 2;     // advance to the second vertex of the next tri strip
              odd = 0;    // next triangle starts with odd == 1 again
              DP_ASSERT( ( count <= i ) || ( ( oldIndices[i-1] != pri ) && ( oldIndices[i] != pri ) ) );
            }
            else if ( odd )
            {
              newIndices.push_back( oldIndices[i-2] );
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[i] );
            }
            else
            {
              newIndices.push_back( oldIndices[i-2] );
              newIndices.push_back( oldIndices[i] );
              newIndices.push_back( oldIndices[i-1] );
            }
          }
        }
        else
        {
          for ( unsigned int i=2, odd=1 ; i<count ; i++, odd ^= 1 )
          {
            if ( odd )
            {
              newIndices.push_back( i-2 );
              newIndices.push_back( i-1 );
              newIndices.push_back( i );
            }
            else
            {
              newIndices.push_back( i-2 );
              newIndices.push_back( i );
              newIndices.push_back( i-1 );
            }
          }
        }
        return( PrimitiveType::TRIANGLES );
      }

      PrimitiveType convertTriangleFan( Primitive * p, vector<unsigned int> & newIndices )
      {
        unsigned int count = p->getElementCount();
        DP_ASSERT( 2 < count );
        newIndices.reserve( 3 * ( count - 2 ) );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( ( oldIndices[0] != pri ) && ( oldIndices[1] != pri ) );
          for ( unsigned int i=2, startI=0 ; i<count ; i++ )
          {
            if ( oldIndices[i] == pri )
            {
              startI = i+1;   // start at next vertex
              i += 2;         // advance to the second vertex of the next tri fan
              DP_ASSERT( ( count <= i ) || ( ( oldIndices[i-1] != pri ) && ( oldIndices[i] != pri ) ) );
            }
            else
            {
              newIndices.push_back( oldIndices[startI] );
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[i] );
            }
          }
        }
        else
        {
          for ( unsigned int i=2 ; i<count ; i++ )
          {
            newIndices.push_back( 0 );
            newIndices.push_back( i-1 );
            newIndices.push_back( i );
          }
        }
        return( PrimitiveType::TRIANGLES );
      }

      PrimitiveType convertQuadStrip( Primitive * p, vector<unsigned int> & newIndices )
      {
        unsigned int count = p->getElementCount();
        DP_ASSERT( 3 < count );
        newIndices.reserve( 4 * ( count / 2 - 1 ) );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( ( oldIndices[0] != pri ) && ( oldIndices[1] != pri ) );
          for ( unsigned int i=3 ; i<count ; i+=2 )
          {
            DP_ASSERT( oldIndices[i-1] != pri );
            if ( oldIndices[i] == pri )
            {
              i += 2;     // advance to the second vertex of the next quad strip
              DP_ASSERT( ( count <= i ) || ( ( oldIndices[i-1] != pri ) && ( oldIndices[i] != pri ) ) );
            }
            else
            {
              newIndices.push_back( oldIndices[i-3] );
              newIndices.push_back( oldIndices[i-2] );
              newIndices.push_back( oldIndices[i] );
              newIndices.push_back( oldIndices[i-1] );
            }
          }
        }
        else
        {
          for ( unsigned int i=3 ; i<count ; i+=2 )
          {
            newIndices.push_back( i-3 );
            newIndices.push_back( i-2 );
            newIndices.push_back( i );
            newIndices.push_back( i-1 );
          }
        }
        return( PrimitiveType::QUADS );
      }

      PrimitiveType convertTriangleStripAdjacency( Primitive * p, vector<unsigned int> & newIndices )
      {
        /*  Indexing of triangles with adjacency:
                         2 - - - 3 - - - 4     8 - - - 9 - - - 10
                                 ^\                    ^\
                           \     | \     |       \     | \     |
                                 |  \                  |  \
                             \   |   \   |         \   |   \   |
                                 |    \                |    \
                               \ |     \ |           \ |     \ |
                                 |      v              |      v
                                 1<------5             7<------11

                                   \     |               \     |

                                     \   |                 \   |

                                       \ |                   \ |

                                         6                     12
        */
        /*  Indexing of triangle strips with adjacency
                                        6                     6

                                        | \                   | \

                                        |   \                 |   \

                                        |     \               |     \

            2 - - - 3- - - >6   2 - - - 3------>7     2 - - - 3------>7- - - 10
                    ^\                  ^^      |             ^^      ^^      |
              \     | \     |     \     | \     | \     \     | \     | \
                    |  \                |  \    |             |  \    |  \    |
                \   |   \   |       \   |   \   |   \     \   |   \   |   \
                    |    \              |    \  |             |    \  |    \  |
                  \ |     \ |         \ |     \ |     \     \ |     \ |     \
                    |      v            |      vv             |      vv      v|
                    1<------5           1<------5 - - - 8     1<------5<------9

                      \     |             \     |               \     | \     |

                        \   |               \   |                 \   |   \   |

                          \ |                 \ |                   \ |     \ |

                            4                   4                     4       8


                                         6       10

                                         | \     | \

                                         |   \   |   \

                                         |     \ |     \
                                 2 - - - 3------>7------>11
                                         ^^      ^^      |
                                   \     | \     | \     | \
                                         |  \    |  \    |
                                     \   |   \   |   \   |   \
                                         |    \  |    \  |
                                       \ |     \ |     \ |     \
                                         |      vv      vv
                                         1<------5<------9 - - - 12

                                           \     | \     |

                                             \   |   \   |

                                               \ |     \ |

                                                 4       8
        */
        /*  Triangles generated by triangle strips with adjacency.
                                       primitive          adjacent
                                       vertices           vertices
            primitive               1st   2nd   3rd     1/2  2/3  3/1
            ---------------        ----  ----  ----    ---- ---- ----
            only (i==0, n==1)        1     3     5       2    6    4
            first (i==0)             1     3     5       2    7    4
            middle (i odd)         2i+3  2i+1  2i+5    2i-1 2i+4 2i+7
            middle (i even)        2i+1  2i+3  2i+5    2i-1 2i+7 2i+4
            last (i==n-1, i odd)   2i+3  2i+1  2i+5    2i-1 2i+4 2i+6
            last (i==n-1, i even)  2i+1  2i+3  2i+5    2i-1 2i+6 2i+4

            Each triangle is drawn using the vertices in the "1st", "2nd", and "3rd"
            columns under "primitive vertices", in that order.  The vertices in the
            "1/2", "2/3", and "3/1" columns under "adjacent vertices" are considered
            adjacent to the edges from the first to the second, from the second to
            the third, and from the third to the first vertex of the triangle,
            respectively.  The six rows correspond to the six cases:  the first and
            only triangle (i=0, n=1), the first triangle of several (i=0, n>0),
            "odd" middle triangles (i=1,3,5...), "even" middle triangles
            (i=2,4,6,...), and special cases for the last triangle inside the
            Begin/End, when i is either even or odd.  For the purposes of this
            table, the first vertex specified after Begin is numbered "1" and the
            first triangle is numbered "0".
        */
        unsigned int count = p->getElementCount();
        DP_ASSERT( 5 < count );
        newIndices.reserve( 6 * ( count / 2 - 2 ) );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( ( oldIndices[0] != pri ) && ( oldIndices[1] != pri ) && ( oldIndices[2] != pri ) && ( oldIndices[3] != pri ) );
          for ( unsigned int i=5, startI=0, odd=1 ; i<count ; i+=2, odd ^= 1 )
          {
            DP_ASSERT( ( oldIndices[i-1] != pri ) && ( oldIndices[i] != pri ) );
            if ( i == startI + 5 )
            { // first triangles need special handling
              newIndices.push_back( oldIndices[i-5] );
              newIndices.push_back( oldIndices[i-4] );
              newIndices.push_back( oldIndices[i-3] );
              if ( ( count <= i+1 ) || ( oldIndices[i+1] == pri ) )
              { // it's the only triangle in the strip
                newIndices.push_back( oldIndices[i] );
              }
              else
              { // it's just the first triangle in the strip
                newIndices.push_back( oldIndices[i+1] );
              }
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[i-2] );
            }
            else if ( ( count <= i+1 ) || ( oldIndices[i+1] == pri ) )
            { // last triangles need special handling
              if ( odd )
              {
                newIndices.push_back( oldIndices[i-5] );
                newIndices.push_back( oldIndices[i-7] );
                newIndices.push_back( oldIndices[i-3] );
                newIndices.push_back( oldIndices[i] );
                newIndices.push_back( oldIndices[i-1] );
                newIndices.push_back( oldIndices[i-2] );
              }
              else
              {
                newIndices.push_back( oldIndices[i-5] );
                newIndices.push_back( oldIndices[i-2] );
                newIndices.push_back( oldIndices[i-1] );
                newIndices.push_back( oldIndices[i] );
                newIndices.push_back( oldIndices[i-3] );
                newIndices.push_back( oldIndices[i-7] );
              }
            }
            else
            {
              DP_ASSERT( i+1<count );
              if ( odd )
              {
                newIndices.push_back( oldIndices[i-5] );
                newIndices.push_back( oldIndices[i-7] );
                newIndices.push_back( oldIndices[i-3] );
                newIndices.push_back( oldIndices[i+1] );
                newIndices.push_back( oldIndices[i-1] );
                newIndices.push_back( oldIndices[i-2] );
              }
              else
              {
                newIndices.push_back( oldIndices[i-5] );
                newIndices.push_back( oldIndices[i-2] );
                newIndices.push_back( oldIndices[i-1] );
                newIndices.push_back( oldIndices[i+1] );
                newIndices.push_back( oldIndices[i-3] );
                newIndices.push_back( oldIndices[i-7] );
              }
            }
            if ( ( i+1<count ) && ( oldIndices[i+1] == pri ) )
            {
              startI = i+2;     // first vertex of next triangle
              i += startI + 3;  // together with i+=2 above, we're at the last vertex of the next triangle
              odd = 0;          // next triangle starts with odd == 1 again
              DP_ASSERT( ( count <= i ) || ( ( oldIndices[i-3] != pri ) && ( oldIndices[i-2] != pri ) && ( oldIndices[i-1] != pri ) && ( oldIndices[i] != pri ) ) );
            }
          }
        }
        else
        {
          // first triangle in the strip
          newIndices.push_back( 0 );
          newIndices.push_back( 1 );
          newIndices.push_back( 2 );
          newIndices.push_back( ( count == 6 ) ? 5 : 6 );
          newIndices.push_back( 4 );
          newIndices.push_back( 3 );
          for ( unsigned int i=7, odd=0 ; i<count ; i+=2, odd ^= 1 )
          {
            if ( count == i+1 )
            { // last triangles need special handling
              if ( odd )
              {
                newIndices.push_back( i-5 );
                newIndices.push_back( i-7 );
                newIndices.push_back( i-3 );
                newIndices.push_back( i );
                newIndices.push_back( i-1 );
                newIndices.push_back( i-2 );
              }
              else
              {
                newIndices.push_back( i-5 );
                newIndices.push_back( i-2 );
                newIndices.push_back( i-1 );
                newIndices.push_back( i );
                newIndices.push_back( i-3 );
                newIndices.push_back( i-7 );
              }
            }
            else if ( odd )
            {
              newIndices.push_back( i-5 );
              newIndices.push_back( i-7 );
              newIndices.push_back( i-3 );
              newIndices.push_back( i+1 );
              newIndices.push_back( i-1 );
              newIndices.push_back( i-2 );
            }
            else
            {
              newIndices.push_back( i-5 );
              newIndices.push_back( i-2 );
              newIndices.push_back( i-1 );
              newIndices.push_back( i+1 );
              newIndices.push_back( i-3 );
              newIndices.push_back( i-7 );
            }
          }
        }
        return( PrimitiveType::TRIANGLES_ADJACENCY );
      }

      PrimitiveType convertLineStripAdajcency( Primitive * p, vector<unsigned int> & newIndices )
      {
        /*
              1 - - - 2----->3 - - - 4     1 - - - 2--->3--->4--->5 - - - 6

              5 - - - 6----->7 - - - 8

                     (a)                             (b)

            (a) Lines with adjacency, (b) Line strip with adjacency.
            The vertices connected with solid lines belong to the main primitives;
            the vertices connected by dashed lines are the adjacent vertices that
            may be used in a geometry shader.
        */
        unsigned int count = p->getElementCount();
        DP_ASSERT( 3 < count );
        newIndices.reserve( 2 * ( count - 3 ) );  // might be more than needed, due to pri elements
        if ( p->isIndexed() )
        {
          unsigned int pri = p->getIndexSet()->getPrimitiveRestartIndex();
          IndexSet::ConstIterator<unsigned int> oldIndices( p->getIndexSet(), p->getElementOffset() );
          DP_ASSERT( ( oldIndices[0] != pri ) && ( oldIndices[1] != pri ) && ( oldIndices[2] != pri ) );
          for ( unsigned int i=3 ; i<count ; i++ )
          {
            if ( oldIndices[i] == pri )
            {
              i += 3;    // advance to the second vertex of the next line strip adjacency
              DP_ASSERT( ( count <= i ) || ( ( oldIndices[i-2] != pri ) && ( oldIndices[i-1] != pri ) && ( oldIndices[i] != pri ) ) );
            }
            else
            {
              newIndices.push_back( oldIndices[i-3] );
              newIndices.push_back( oldIndices[i-2] );
              newIndices.push_back( oldIndices[i-1] );
              newIndices.push_back( oldIndices[i] );
            }
          }
        }
        else
        {
          for ( unsigned int i=3 ; i<count ; i++ )
          {
            newIndices.push_back( i-3 );
            newIndices.push_back( i-2 );
            newIndices.push_back( i-1 );
            newIndices.push_back( i );
          }
        }
        return( PrimitiveType::LINES_ADJACENCY );
      }

      void DestrippingTraverser::doApply( const dp::sg::core::NodeSharedPtr & root )
      {
        m_primitiveMap.clear();
        ExclusiveTraverser::doApply( root );
      }

      void DestrippingTraverser::handleGeoNode( GeoNode * p )
      {
        DP_ASSERT( !m_primitive );
        ExclusiveTraverser::handleGeoNode( p );
        if ( m_primitive )
        {
          p->setPrimitive( m_primitive );
          m_primitive.reset();
          setTreeModified();
        }
      }

      void DestrippingTraverser::handlePrimitive( Primitive * p )
      {
        // if the vertices per primitive is 0, p is a destrippable Primitive
        if ( optimizationAllowed( p->getSharedPtr<Primitive>() ) && ( p->getNumberOfVerticesPerPrimitive() == 0 ) )
        {
          std::map<Primitive*,dp::sg::core::PrimitiveSharedPtr>::const_iterator it = m_primitiveMap.find( p );
          if ( it == m_primitiveMap.end() )
          {
            vector<unsigned int> newIndices;
            PrimitiveType newPrimitiveType = PrimitiveType::UNINITIALIZED;
            switch( p->getPrimitiveType() )
            {
              case PrimitiveType::LINE_STRIP :
                newPrimitiveType = convertLineStrip( p, newIndices );
                break;
              case PrimitiveType::LINE_LOOP :
                newPrimitiveType = convertLineLoop( p, newIndices );
                break;
              case PrimitiveType::TRIANGLE_STRIP :
                newPrimitiveType = convertTriangleStrip( p, newIndices );
                break;
              case PrimitiveType::TRIANGLE_FAN :
                newPrimitiveType = convertTriangleFan( p, newIndices );
                break;
              case PrimitiveType::QUAD_STRIP :
                newPrimitiveType = convertQuadStrip( p, newIndices );
                break;
              case PrimitiveType::TRIANGLE_STRIP_ADJACENCY :
                newPrimitiveType = convertTriangleStripAdjacency( p, newIndices );
                break;
              case PrimitiveType::LINE_STRIP_ADJACENCY :
                newPrimitiveType = convertLineStripAdajcency( p, newIndices );
                break;
              default :
                DP_ASSERT( false );
                break;
            }
            DP_ASSERT( newPrimitiveType != PrimitiveType::UNINITIALIZED )
            DP_ASSERT( !newIndices.empty() );
            IndexSetSharedPtr newIndexSet = IndexSet::create();
            if ( p->getIndexSet() )
            {
              *newIndexSet = *p->getIndexSet();
            }
            newIndexSet->setData( &newIndices[0], dp::checked_cast<unsigned int>(newIndices.size()) );
            newIndexSet->setPrimitiveRestartIndex( ~0 );
            PrimitiveSharedPtr primitive = Primitive::create( newPrimitiveType );
            *primitive = *p;
            primitive->setInstanceCount( p->getInstanceCount() );
            primitive->setVertexAttributeSet( p->getVertexAttributeSet() );
            primitive->setIndexSet( newIndexSet );
            it = m_primitiveMap.insert( std::make_pair( p, primitive ) ).first;
          }
          m_primitive = it->second;
        }
      }

      bool DestrippingTraverser::optimizationAllowed( ObjectSharedPtr const& obj )
      {
        DP_ASSERT( obj != NULL );

        return (obj->getHints( Object::DP_SG_HINT_DYNAMIC ) == 0);
      }

    } // namespace algorithm
  } // namespace sp
} // namespace dp
