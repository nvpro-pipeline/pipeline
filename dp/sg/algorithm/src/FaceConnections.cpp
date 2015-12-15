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


#include <dp/sg/algorithm/FaceConnections.h>
#include <dp/sg/core/Primitive.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      FaceConnections::FaceConnections( const dp::sg::core::Primitive * p )
        : m_faceConnections(p->getElementCount(),~0)
        , m_faceConnectionCounts(p->getNumberOfPrimitives())
        , m_faceSets(p->getNumberOfVerticesPerPrimitive()+1)
      {
        DP_ASSERT( ( p->getPrimitiveType() == dp::sg::core::PrimitiveType::TRIANGLES )
                || ( p->getPrimitiveType() == dp::sg::core::PrimitiveType::QUADS ) );

        unsigned int elementCount = p->getElementCount();
        unsigned int primitiveSize = p->getNumberOfVerticesPerPrimitive();
        DP_ASSERT( ( elementCount % primitiveSize ) == 0 );

        //  for each vertex build a list of face indices where it is used
        unsigned int vertexCount = p->getVertexAttributeSet()->getNumberOfVertices();
        std::vector<std::list<unsigned int> > verticesToFaceLists( vertexCount );
        dp::sg::core::IndexSet::ConstIterator<unsigned int> indices( p->getIndexSet(), p->getElementOffset() );
        for ( unsigned int i=0, k=0 ; i<elementCount ; i+=primitiveSize, k++ )
        {
          for ( unsigned int j=0 ; j<primitiveSize ; j++ )
          {
            verticesToFaceLists[indices[i+j]].push_back( k );
          }
        }

        //  for each face and for each not yet connected edge look for a (unique!) face sharing it and mark them as connected
        unsigned int nof = elementCount / primitiveSize;
        for ( unsigned int i=0; i<nof ; i++ )
        {
          for ( unsigned int j=0 ; j<primitiveSize-1 ; j++ )
          {
            connectFaces( indices, i, primitiveSize, j, j+1, verticesToFaceLists );
          }
          connectFaces( indices, i, primitiveSize, primitiveSize-1, 0, verticesToFaceLists );
        }

        //  build a set of zero-, one-, two-, three-, (and four-)connected faces
        for ( unsigned int i=0, k=0 ; i<elementCount ; i+=primitiveSize, k++ )
        {
          unsigned int connectionCount = primitiveSize;
          for ( unsigned int j=0 ; j<primitiveSize ; j++ )
          {
            if ( m_faceConnections[i+j] == 0xFFFFFFFF )
            {
              connectionCount--;
            }
          }
          m_faceSets[connectionCount].insert( k );
          m_faceConnectionCounts[k] = connectionCount;
        }
      }

      void FaceConnections::disconnectFace( unsigned int fi  )
      {
        unsigned int vpf = dp::checked_cast<unsigned int>(m_faceSets.size() - 1);
        for ( unsigned int i=0 ; i<vpf ; i++ )
        {
          unsigned int cfi = m_faceConnections[vpf*fi+i];
          if ( cfi != ~0 )
          {
            m_faceSets[m_faceConnectionCounts[cfi]].erase( cfi );
            unsigned int ce = ~0;
            for ( unsigned int j=0 ; ce==~0 && j<vpf ; j++ )
            {
              if ( m_faceConnections[vpf*cfi+j] == fi )
              {
                ce = j;
              }
            }
            DP_ASSERT( ce != ~0 );
            DP_ASSERT( m_faceConnections[vpf*cfi+ce] == fi );
            m_faceConnections[vpf*cfi+ce] = ~0;
            m_faceConnectionCounts[cfi]--;
            m_faceSets[m_faceConnectionCounts[cfi]].insert( cfi );
          }
        }
        m_faceSets[m_faceConnectionCounts[fi]].erase( fi );
      }

      void FaceConnections::disconnectFaces( const unsigned int * faceIndices, unsigned int faceCount )
      {
        for ( unsigned int i=0 ; i<faceCount ; i++ )
        {
          disconnectFace( faceIndices[i] );
        }
      }

      void FaceConnections::disconnectFaces( const std::list<unsigned int> &faceList )
      {
        for ( std::list<unsigned int>::const_iterator lit=faceList.begin() ; lit!=faceList.end() ; ++lit )
        {
          disconnectFace( *lit );
        }
      }

      void checkQuadStrip( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi, unsigned int le
                         , const std::vector<unsigned int> & faceConnections, std::list<unsigned int> & faceList
                         , std::list<unsigned int> & vertexList )
      {
        std::set<unsigned int> listSet;
        faceList.clear();
        vertexList.clear();

        //  start the strip
        vertexList.push_back( indices[4*fi+(le+2)%4] );
        vertexList.push_back( indices[4*fi+(le+3)%4] );
        vertexList.push_back( indices[4*fi+(le+1)%4] );
        vertexList.push_back( indices[4*fi+le] );
        faceList.push_back( fi );
        listSet.insert( fi );
        unsigned int  ble = ( le + 2 ) % 4; //  leaving edge for backward list

        //  determine the forward list
        while ( fi != 0xFFFFFFFF )
        {
          //  determine the edge where the next face is entered to get the edge where this next face has to be left
          unsigned int nfi = faceConnections[4*fi+le];
          if ( nfi != 0xFFFFFFFF ) 
          {
            if ( listSet.find( nfi ) == listSet.end() )
            {
              //  determine entering and leaving edge for next face
              unsigned int ee = ( faceConnections[4*nfi+0] == fi ) ? 0 : ( faceConnections[4*nfi+1] == fi ) ? 1 : ( faceConnections[4*nfi+2] == fi ) ? 2 : 3;
              DP_ASSERT( faceConnections[4*nfi+ee] == fi );
              //  the leaving edge is entering edge + 2
              le = ( ee + 2 ) % 4;
              //  the vertices opposite to the entering edge are new in the strip
              vertexList.push_back( indices[4*nfi+(ee+3)%4] );
              vertexList.push_back( indices[4*nfi+(ee+2)%4] );
              faceList.push_back( nfi );
              listSet.insert( nfi );
            }
            else
            {
              nfi = 0xFFFFFFFF;
            }
          }
          fi = nfi;
        }

        //  determine the backward list
        fi = faceList.front();
        le = ble;
        unsigned int  bCount = 0;
        while ( fi != 0xFFFFFFFF )
        {
          //  determine the edge where the next face is entered to get the edge where this next face has to be left
          unsigned int nfi = faceConnections[4*fi+le];
          if ( nfi != 0xFFFFFFFF ) 
          {
            if ( listSet.find( nfi ) == listSet.end() )
            {
              //  determine entering and leaving edge for next face
              unsigned int ee = ( faceConnections[4*nfi+0] == fi )
                                ? 0
                                : ( faceConnections[4*nfi+1] == fi )
                                  ? 1
                                  : ( faceConnections[4*nfi+2] == fi )
                                    ? 2
                                    : 3;
              DP_ASSERT( faceConnections[4*nfi+ee] == fi );
              //  the leaving edge is entering edge + 2
              le = ( ee + 2 ) % 4;
              //  the vertices opposite to the entering edge are new in the strip
              vertexList.push_front( indices[4*nfi+(ee+3)%4] );
              vertexList.push_front( indices[4*nfi+(ee+2)%4] );
              faceList.push_front( nfi );
              listSet.insert( nfi );
              bCount++;
            }
            else
            {
              nfi = 0xFFFFFFFF;
            }
          }
          fi = nfi;
        }
      }

      unsigned int FaceConnections::findLongestQuadStrip( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                                        , std::vector<unsigned int> & stripIndices
                                                        , std::list<unsigned int> & stripFaces )
      {
        DP_ASSERT( m_faceSets.size() == 5 );
        std::list<unsigned int> faceList[2];
        std::list<unsigned int> vertexList[2];

        //  determine the face lists and the corresponding strips (only two possible lists here !)
        checkQuadStrip( indices, fi, 0, m_faceConnections, faceList[0], vertexList[0] );
        checkQuadStrip( indices, fi, 1, m_faceConnections, faceList[1], vertexList[1] );

        //  determine the longest list and use it
        unsigned int li = ( faceList[0].size() >= faceList[1].size() ) ? 0 : 1;

        copy( vertexList[li].begin(), vertexList[li].end(), back_inserter(stripIndices) );
        stripFaces.swap( faceList[li] );
        return( dp::checked_cast<unsigned int>(stripFaces.size()) );
      }

      void checkTriStrip( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi, unsigned int le
                        , const std::vector<unsigned int> & faceConnections, std::list<unsigned int> & faceList
                        , std::list<unsigned int> & vertexList )
      {
        std::set<unsigned int> listSet;
        faceList.clear();
        vertexList.clear();

        //  start the strip
        vertexList.push_back( indices[3*fi+(le+2)%3] );
        vertexList.push_back( indices[3*fi+le] );
        vertexList.push_back( indices[3*fi+(le+1)%3] );
        faceList.push_back( fi );
        listSet.insert( fi );
        unsigned int  ble = ( le + 2 ) % 3; //  leaving edge for backward list

        //  determine the forward list
        while ( fi != 0xFFFFFFFF )
        {
          //  determine the edge where the next face is entered to get the edge where this next face has to be left
          unsigned int nfi = faceConnections[3*fi+le];
          if ( nfi != 0xFFFFFFFF ) 
          {
            if ( listSet.find( nfi ) == listSet.end() )
            {
              //  determine entering and leaving edge for next face
              unsigned int ee = ( faceConnections[3*nfi+0] == fi ) ? 0 : ( faceConnections[3*nfi+1] == fi ) ? 1 : 2;
              DP_ASSERT( faceConnections[3*nfi+ee] == fi );
              //  when there are odd elements in the forward list, the leaving edge is entering edge + 2
              //  otherwise it's the entering edge + 1
              le = ( ee + 1 + ( dp::checked_cast<unsigned int>(faceList.size()) % 2 ) ) % 3;
              //  the vertex not on the entering edge is new in the strip
              vertexList.push_back( indices[3*nfi+( ee + 2 ) % 3] );
              faceList.push_back( nfi );
              listSet.insert( nfi );
            }
            else
            {
              nfi = 0xFFFFFFFF;
            }
          }
          fi = nfi;
        }

        //  determine the backward list
        fi = faceList.front();
        le = ble;
        unsigned int  bCount = 0;
        while ( fi != 0xFFFFFFFF )
        {
          //  determine the edge where the next face is entered to get the edge where this next face has to be left
          unsigned int nfi = faceConnections[3*fi+le];
          if ( nfi != 0xFFFFFFFF ) 
          {
            if ( listSet.find( nfi ) == listSet.end() )
            {
              //  determine entering and leaving edge for next face
              unsigned int ee = ( faceConnections[3*nfi+0] == fi ) ? 0 : ( faceConnections[3*nfi+1] == fi ) ? 1 : 2;
              DP_ASSERT( faceConnections[3*nfi+ee] == fi );
              //  when there are odd elements in the backward list, the leaving edge is entering edge + 2
              //  otherwise it's the entering edge + 1
              le = ( ee + 1 + ( bCount % 2 ) ) % 3;
              //  the vertex not on the entering edge is new in the strip
              vertexList.push_front( indices[3*nfi+(ee+2)%3] );
              faceList.push_front( nfi );
              listSet.insert( nfi );
              bCount++;
            }
            else
            {
              nfi = 0xFFFFFFFF;
            }
          }
          fi = nfi;
        }
        //  the backward list has to have even members, otherwise delete the front element
        if ( bCount % 2 )
        {
          vertexList.pop_front();
          faceList.pop_front();
        }
      }

      unsigned int FaceConnections::findLongestTriStrip( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                                       , std::vector<unsigned int> & stripIndices
                                                       , std::list<unsigned int> & stripFaces )
      {
        DP_ASSERT( m_faceSets.size() == 4 );
        std::list<unsigned int> faceList[3];
        std::list<unsigned int> vertexList[3];

        //  determine the face lists and the corresponding strips
        checkTriStrip( indices, fi, 0, m_faceConnections, faceList[0], vertexList[0] );
        checkTriStrip( indices, fi, 1, m_faceConnections, faceList[1], vertexList[1] );
        checkTriStrip( indices, fi, 2, m_faceConnections, faceList[2], vertexList[2] );

        //  determine the longest list and use it
        unsigned int li = ( faceList[0].size() >= faceList[1].size() )
                          ? ( faceList[0].size() >= faceList[2].size() ) ? 0 : 2
                          : ( faceList[1].size() >= faceList[2].size() ) ? 1 : 2;

        copy( vertexList[li].begin(), vertexList[li].end(), back_inserter(stripIndices) );
        stripFaces.swap( faceList[li] );
        return( dp::checked_cast<unsigned int>(stripFaces.size()) );
      }

      unsigned int findEdge( unsigned int fromFace, unsigned toFace, unsigned int ps
                           , const std::vector<unsigned int> & faceConnections )
      {
        unsigned int edge = ~0;
        unsigned int vi = ps * fromFace;
        for ( unsigned int i=0 ; i<ps ; i++ )
        {
          if ( faceConnections[vi+i] == toFace )
          {
            edge = i;
            break;
          }
        }
        return( edge );
      }

    // For all those checks in checkQuadPatch4x4Startxy, I assume the QuadPatch is formed out of 9 quads,
    // ordered like that:
    //    _____________
    //    |   |   |   |
    //    | 6 | 7 | 8 |
    //    |___|___|___|
    //    |   |   |   |
    //    | 3 | 4 | 5 |
    //    |___|___|___|
    //    |   |   |   |
    //    | 0 | 1 | 2 |
    //    |___|___|___|

      bool checkQuadPatch4x4Start0i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi, unsigned int ei01
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int pf[9] )
      {
        bool ok = false;
        pf[0] = fi;
        pf[1] = faceConnections[4*pf[0]+ei01];
        DP_ASSERT( pf[1] != pf[0] );
        if ( pf[1] != ~0 )
        {
          unsigned int ei10 = findEdge( pf[1], pf[0], 4, faceConnections );
          DP_ASSERT( ei10 != ~0 );
          pf[2] = faceConnections[4*pf[1]+(ei10+2)%4];
          DP_ASSERT( pf[2] != pf[1] );
          if ( ( pf[2] != ~0 ) && ( pf[2] != pf[0] ) )
          {
            unsigned int ei21 = findEdge( pf[2], pf[1], 4, faceConnections );
            DP_ASSERT( ei21 != ~0 );
            pf[3] = faceConnections[4*pf[0]+(ei01+1)%4];
            DP_ASSERT( pf[3] != pf[0] );
            if ( ( pf[3] != ~0 ) && ( pf[3] != pf[2] ) && ( pf[3] != pf[1] ) )
            {
              unsigned int ei30 = findEdge( pf[3], pf[0], 4, faceConnections );
              DP_ASSERT( ei30 != ~0 );
              pf[4] = faceConnections[4*pf[3]+(ei30+1)%4];
              DP_ASSERT( pf[4] != pf[3] );
              if ( ( pf[4] != ~0 ) && ( pf[4] != pf[2] ) && ( pf[4] != pf[1] ) && ( pf[4] != pf[0] ) )
              {
                unsigned int ei43 = findEdge( pf[4], pf[3], 4, faceConnections );
                DP_ASSERT( ei43 != ~0 );
                if ( faceConnections[4*pf[4]+(ei43+1)%4] == pf[1] )
                {
                  pf[5] = faceConnections[4*pf[4]+(ei43+2)%4];
                  DP_ASSERT( pf[5] != pf[4] );
                  if ( ( pf[5] != ~0 ) && ( pf[5] != pf[3] ) && ( pf[5] != pf[2] ) && ( pf[5] != pf[1] ) && ( pf[5] != pf[0] ) )
                  {
                    unsigned int ei54 = findEdge( pf[5], pf[4], 4, faceConnections );
                    DP_ASSERT( ei54 != ~0 );
                    if ( faceConnections[4*pf[5]+(ei54+1)%4] == pf[2] )
                    {
                      pf[6] = faceConnections[4*pf[3]+(ei30+2)%4];
                      DP_ASSERT( pf[6] != pf[3] );
                      if ( ( pf[6] != ~0 ) && ( pf[6] != pf[5] ) && ( pf[6] != pf[4] ) && ( pf[6] != pf[2] ) && ( pf[6] != pf[1] ) && ( pf[6] != pf[0] ) )
                      {
                        unsigned int ei63 = findEdge( pf[6], pf[3], 4, faceConnections );
                        DP_ASSERT( ei63 != ~0 );
                        pf[7] = faceConnections[4*pf[6]+(ei63+1)%4];
                        DP_ASSERT( pf[7] != pf[6] );
                        if ( ( pf[7] != ~0 ) && ( pf[7] != pf[5] ) && ( pf[7] != pf[4] ) && ( pf[7] != pf[3] ) && ( pf[7] != pf[2] ) && ( pf[7] != pf[1] ) && ( pf[7] != pf[0] ) )
                        {
                          unsigned int ei76 = findEdge( pf[7], pf[6], 4, faceConnections );
                          DP_ASSERT( ei76 != ~0 );
                          if ( faceConnections[4*pf[7]+(ei76+1)%4] == pf[4] )
                          {
                            pf[8] = faceConnections[4*pf[7]+(ei76+2)%4];
                            DP_ASSERT( pf[8] != pf[7] );
                            if ( ( pf[8] != ~0 ) && ( pf[8] != pf[6] ) && ( pf[8] != pf[5] ) && ( pf[8] != pf[4] ) && ( pf[8] != pf[3] ) && ( pf[8] != pf[2] ) && ( pf[8] != pf[1] ) && ( pf[8] != pf[0] ) )
                            {
                              unsigned int ei87 = findEdge( pf[8], pf[7], 4, faceConnections );
                              DP_ASSERT( ei87 != ~0 );
                              if ( faceConnections[4*pf[8]+(ei87+1)%4] == pf[5] )
                              {
                                patchIndices.push_back( indices[4*pf[0]+(ei01+3)%4] );
                                DP_ASSERT( indices[4*pf[0]+ei01] == indices[4*pf[1]+(ei10+1)%4] );
                                patchIndices.push_back( indices[4*pf[0]+ei01] );
                                DP_ASSERT( indices[4*pf[1]+(ei10+2)%4] == indices[4*pf[2]+(ei21+1)%4] );
                                patchIndices.push_back( indices[4*pf[1]+(ei10+2)%4] );
                                patchIndices.push_back( indices[4*pf[2]+(ei21+2)%4] );
                                DP_ASSERT( indices[4*pf[0]+(ei01+2)%4] == indices[4*pf[3]+ei30] );
                                patchIndices.push_back( indices[4*pf[0]+(ei01+2)%4] );
                                DP_ASSERT( indices[4*pf[0]+(ei01+1)%4] == indices[4*pf[1]+ei10] );
                                DP_ASSERT( indices[4*pf[0]+(ei01+1)%4] == indices[4*pf[3]+(ei30+1)%4] );
                                DP_ASSERT( indices[4*pf[0]+(ei01+1)%4] == indices[4*pf[4]+(ei43+1)%4] );
                                patchIndices.push_back( indices[4*pf[0]+(ei01+1)%4] );
                                DP_ASSERT( indices[4*pf[1]+(ei10+3)%4] == indices[4*pf[2]+ei21] );
                                DP_ASSERT( indices[4*pf[1]+(ei10+3)%4] == indices[4*pf[4]+(ei43+2)%4] );
                                DP_ASSERT( indices[4*pf[1]+(ei10+3)%4] == indices[4*pf[5]+(ei54+1)%4] );
                                patchIndices.push_back( indices[4*pf[1]+(ei10+3)%4] );
                                DP_ASSERT( indices[4*pf[2]+(ei21+3)%4] == indices[4*pf[5]+(ei54+2)%4] );
                                patchIndices.push_back( indices[4*pf[2]+(ei21+3)%4] );
                                DP_ASSERT( indices[4*pf[3]+(ei30+3)%4] == indices[4*pf[6]+ei63] );
                                patchIndices.push_back( indices[4*pf[3]+(ei30+3)%4] );
                                DP_ASSERT( indices[4*pf[3]+(ei30+2)%4] == indices[4*pf[4]+ei43] );
                                DP_ASSERT( indices[4*pf[3]+(ei30+2)%4] == indices[4*pf[6]+(ei63+1)%4] );
                                DP_ASSERT( indices[4*pf[3]+(ei30+2)%4] == indices[4*pf[7]+(ei76+1)%4] );
                                patchIndices.push_back( indices[4*pf[3]+(ei30+2)%4] );
                                DP_ASSERT( indices[4*pf[4]+(ei43+3)%4] == indices[4*pf[5]+ei54] );
                                DP_ASSERT( indices[4*pf[4]+(ei43+3)%4] == indices[4*pf[7]+(ei76+2)%4] );
                                DP_ASSERT( indices[4*pf[4]+(ei43+3)%4] == indices[4*pf[8]+(ei87+1)%4] );
                                patchIndices.push_back( indices[4*pf[4]+(ei43+3)%4] );
                                DP_ASSERT( indices[4*pf[5]+(ei54+3)%4] == indices[4*pf[8]+(ei87+2)%4] );
                                patchIndices.push_back( indices[4*pf[5]+(ei54+3)%4] );
                                patchIndices.push_back( indices[4*pf[6]+(ei63+3)%4] );
                                DP_ASSERT( indices[4*pf[6]+(ei63+2)%4] == indices[4*pf[7]+ei76] );
                                patchIndices.push_back( indices[4*pf[6]+(ei63+2)%4] );
                                DP_ASSERT( indices[4*pf[7]+(ei76+3)%4] == indices[4*pf[8]+ei87] );
                                patchIndices.push_back( indices[4*pf[7]+(ei76+3)%4] );
                                patchIndices.push_back( indices[4*pf[8]+(ei87+3)%4] );
                                ok = true;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start0( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start0i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start1i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi1, unsigned int ei10
                                    , const std::vector<unsigned int> & faceConnections
                                    , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi0 = faceConnections[4*fi1+ei10];
        if ( fi0 != ~0 )
        {
          unsigned int ei01 = findEdge( fi0, fi1, 4, faceConnections );
          DP_ASSERT( ei01 != ~0 );
          ok = checkQuadPatch4x4Start0i( indices, fi0, ei01, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start1( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start1i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start2i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi2, unsigned int ei21
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi1 = faceConnections[4*fi2+ei21];
        if ( fi1 != ~0 )
        {
          unsigned int ei12 = findEdge( fi1, fi2, 4, faceConnections );
          DP_ASSERT( ei12 != ~0 );
          ok = checkQuadPatch4x4Start1i( indices, fi1, (ei12+2)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start2( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start2i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start3i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi3, unsigned int ei30
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi0 = faceConnections[4*fi3+ei30];
        if ( fi0 != ~0 )
        {
          unsigned int ei03 = findEdge( fi0, fi3, 4, faceConnections );
          DP_ASSERT( ei03 != ~0 );
          ok = checkQuadPatch4x4Start0i( indices, fi0, (ei03+3)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start3( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start3i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start4i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi4, unsigned int ei43
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi3 = faceConnections[4*fi4+ei43];
        if ( fi3 != ~0 )
        {
          unsigned int ei34 = findEdge( fi3, fi4, 4, faceConnections );
          DP_ASSERT( ei34 != ~0 );
          ok = checkQuadPatch4x4Start3i( indices, fi3, (ei34+3)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start4( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start4i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start5i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi5, unsigned int ei54
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi4 = faceConnections[4*fi5+ei54];
        if ( fi4 != ~0 )
        {
          unsigned int ei45 = findEdge( fi4, fi5, 4, faceConnections );
          DP_ASSERT( ei45 != ~0 );
          ok = checkQuadPatch4x4Start4i( indices, fi4, (ei45+2)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start5( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start5i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start6i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi6, unsigned int ei63
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi3 = faceConnections[4*fi6+ei63];
        if ( fi3 != ~0 )
        {
          unsigned int ei36 = findEdge( fi3, fi6, 4, faceConnections );
          DP_ASSERT( ei36 != ~0 );
          ok = checkQuadPatch4x4Start3i( indices, fi3, (ei36+2)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start6( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start6i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start7i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi7, unsigned int ei76
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi6 = faceConnections[4*fi7+ei76];
        if ( fi6 != ~0 )
        {
          unsigned int ei67 = findEdge( fi6, fi7, 4, faceConnections );
          DP_ASSERT( ei67 != ~0 );
          ok = checkQuadPatch4x4Start6i( indices, fi6, (ei67+3)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start7( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start7i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start8i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi8, unsigned int ei87
                                   , const std::vector<unsigned int> & faceConnections
                                   , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi7 = faceConnections[4*fi8+ei87];
        if ( fi7 != ~0 )
        {
          unsigned int ei78 = findEdge( fi7, fi8, 4, faceConnections );
          DP_ASSERT( ei78 != ~0 );
          ok = checkQuadPatch4x4Start7i( indices, fi7, (ei78+2)%4, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkQuadPatch4x4Start8( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                  , const std::vector<unsigned int> & faceConnections
                                  , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<4 && !ok ; i++ )
        {
          ok = checkQuadPatch4x4Start8i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool FaceConnections::findQuadPatch4x4( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                            , std::vector<unsigned int> & patchIndices
                                            , unsigned int patchFaces[9] )
      {
        DP_ASSERT( m_faceSets.size() == 5 );
        return(   checkQuadPatch4x4Start0( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start1( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start2( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start3( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start4( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start5( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start6( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start7( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkQuadPatch4x4Start8( indices, fi, m_faceConnections, patchIndices, patchFaces ) );
      }

    // For all those checks in checkTriPatch4Startxy, I assume the TriPatch is formed out of 9 triangles, ordered
    // like that:
    //               /\
    //              /  \
    //             /  8 \
    //            /______\
    //           /\      /\
    //          /  \  6 /  \
    //         /  5 \  /  7 \
    //        /______\/______\
    //       /\      /\      /\
    //      /  \  1 /  \  3 /  \
    //     /  0 \  /  2 \  /  4 \
    //    /______\/______\/______\

      bool checkTriPatch4Start0i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi, unsigned int ei01
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int pf[9] )
      {
        bool ok = false;
        pf[0] = fi;
        pf[1] = faceConnections[3*pf[0]+ei01];
        DP_ASSERT( pf[1] != pf[0] );
        if ( pf[1] != ~0 )
        {
          unsigned int ei10 = findEdge( pf[1], pf[0], 3, faceConnections );
          DP_ASSERT( ei10 != ~0 );
          pf[2] = faceConnections[3*pf[1]+(ei10+1)%3];
          DP_ASSERT( pf[2] != pf[1] );
          if ( ( pf[2] != ~0 ) && ( pf[2] != pf[0] ) )
          {
            unsigned int ei21 = findEdge( pf[2], pf[1], 3, faceConnections );
            DP_ASSERT( ei21 != ~0 );
            pf[3] = faceConnections[3*pf[2]+(ei21+2)%3];
            DP_ASSERT( pf[3] != pf[2] );
            if ( ( pf[3] != ~0 ) && ( pf[3] != pf[1] ) && ( pf[3] != pf[0] ) )
            {
              unsigned int ei32 = findEdge( pf[3], pf[2], 3, faceConnections );
              DP_ASSERT( ei32 != ~0 );
              pf[4] = faceConnections[3*pf[3]+(ei32+1)%3];
              DP_ASSERT( pf[4] != pf[3] );
              if ( ( pf[4] != ~0 ) && ( pf[4] != pf[2] ) && ( pf[4] != pf[1] ) && ( pf[4] != pf[0] ) )
              {
                unsigned int ei43 = findEdge( pf[4], pf[3], 3, faceConnections );
                DP_ASSERT( ei43 != ~0 );
                pf[5] = faceConnections[3*pf[1]+(ei10+2)%3];
                DP_ASSERT( pf[5] != pf[1] );
                if ( ( pf[5] != ~0 ) && ( pf[5] != pf[4] ) && ( pf[5] != pf[3] ) && ( pf[5] != pf[2] ) && ( pf[5] != pf[0] ) )
                {
                  unsigned int ei51 = findEdge( pf[5], pf[1], 3, faceConnections );
                  DP_ASSERT( ei51 != ~0 );
                  pf[6] = faceConnections[3*pf[5]+(ei51+1)%3];
                  DP_ASSERT( pf[6] != pf[5] );
                  if ( ( pf[6] != ~0 ) && ( pf[6] != pf[4] ) && ( pf[6] != pf[3] ) && ( pf[6] != pf[2] ) && ( pf[6] != pf[1] ) && ( pf[6] != pf[0] ) )
                  {
                    unsigned int ei65 = findEdge( pf[6], pf[5], 3, faceConnections );
                    DP_ASSERT( ei65 != ~0 );
                    pf[7] = faceConnections[3*pf[6]+(ei65+1)%3];
                    DP_ASSERT( pf[7] != pf[6] );
                    if ( ( pf[7] != ~0 ) && ( pf[7] != pf[5] ) && ( pf[7] != pf[4] ) && ( pf[7] != pf[3] ) && ( pf[7] != pf[2] ) && ( pf[7] != pf[1] ) && ( pf[7] != pf[0] ) )
                    {
                      unsigned int ei76 = findEdge( pf[7], pf[6], 3, faceConnections );
                      DP_ASSERT( ei76 != ~0 );
                      if ( faceConnections[3*pf[7]+(ei76+1)%3] == pf[3] )
                      {
                        pf[8] = faceConnections[3*pf[6]+(ei65+2)%3];
                        DP_ASSERT( pf[8] != pf[6] );
                        if ( ( pf[8] != ~0 ) && ( pf[8] != pf[7] ) && ( pf[8] != pf[5] ) && ( pf[8] != pf[4] ) && ( pf[8] != pf[3] ) && ( pf[8] != pf[2] ) && ( pf[8] != pf[1] ) && ( pf[8] != pf[0] ) )
                        {
                          unsigned int ei86 = findEdge( pf[8], pf[6], 3, faceConnections );
                          DP_ASSERT( ei86 != ~0 );
                          patchIndices.push_back( indices[3*pf[0]+(ei01+2)%3] );
                          DP_ASSERT( indices[3*pf[0]+ei01] == indices[3*pf[1]+(ei10+1)%3] );
                          DP_ASSERT( indices[3*pf[0]+ei01] == indices[3*pf[2]+(ei21+1)%3] );
                          patchIndices.push_back( indices[3*pf[0]+ei01] );
                          DP_ASSERT( indices[3*pf[2]+(ei21+2)%3] == indices[3*pf[3]+(ei32+1)%3] );
                          DP_ASSERT( indices[3*pf[2]+(ei21+2)%3] == indices[3*pf[4]+(ei43+1)%3] );
                          patchIndices.push_back( indices[3*pf[2]+(ei21+2)%3] );
                          patchIndices.push_back( indices[3*pf[4]+(ei43+2)%3] );
                          DP_ASSERT( indices[3*pf[0]+(ei01+1)%3] == indices[3*pf[1]+ei10] );
                          DP_ASSERT( indices[3*pf[0]+(ei01+1)%3] == indices[3*pf[5]+ei51] );
                          patchIndices.push_back( indices[3*pf[0]+(ei01+1)%3] );
                          DP_ASSERT( indices[3*pf[1]+(ei10+2)%3] == indices[3*pf[2]+ei21] );
                          DP_ASSERT( indices[3*pf[1]+(ei10+2)%3] == indices[3*pf[3]+ei32] );
                          DP_ASSERT( indices[3*pf[1]+(ei10+2)%3] == indices[3*pf[5]+(ei51+1)%3] );
                          DP_ASSERT( indices[3*pf[1]+(ei10+2)%3] == indices[3*pf[6]+(ei65+1)%3] );
                          DP_ASSERT( indices[3*pf[1]+(ei10+2)%3] == indices[3*pf[7]+(ei76+1)%3] );
                          patchIndices.push_back( indices[3*pf[1]+(ei10+2)%3] );
                          DP_ASSERT( indices[3*pf[3]+(ei32+2)%3] == indices[3*pf[4]+ei43] );
                          DP_ASSERT( indices[3*pf[3]+(ei32+2)%3] == indices[3*pf[7]+(ei76+2)%3] );
                          patchIndices.push_back( indices[3*pf[3]+(ei32+2)%3] );
                          DP_ASSERT( indices[3*pf[5]+(ei51+2)%3] == indices[3*pf[6]+ei65] );
                          DP_ASSERT( indices[3*pf[5]+(ei51+2)%3] == indices[3*pf[8]+ei86] );
                          patchIndices.push_back( indices[3*pf[5]+(ei51+2)%3] );
                          DP_ASSERT( indices[3*pf[6]+(ei65+2)%3] == indices[3*pf[7]+ei76] );
                          DP_ASSERT( indices[3*pf[6]+(ei65+2)%3] == indices[3*pf[8]+(ei86+1)%3] );
                          patchIndices.push_back( indices[3*pf[6]+(ei65+2)%3] );
                          patchIndices.push_back( indices[3*pf[8]+(ei86+2)%3] );
                          ok = true;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return( ok );
      }

      bool checkTriPatch4Start0( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start0i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start1i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi1, unsigned int ei10
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi0 = faceConnections[3*fi1+ei10];
        if ( fi0 != ~0 )
        {
          unsigned int ei01 = findEdge( fi0, fi1, 3, faceConnections );
          DP_ASSERT( ei01 != ~0 );
          ok = checkTriPatch4Start0i( indices, fi0, ei01, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start1( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start1i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start2i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi2, unsigned int ei21
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi1 = faceConnections[3*fi2+ei21];
        if ( fi1 != ~0 )
        {
          unsigned int ei12 = findEdge( fi1, fi2, 3, faceConnections );
          DP_ASSERT( ei12 != ~0 );
          ok = checkTriPatch4Start1i( indices, fi1, (ei12+2)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start2( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start2i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start3i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi3, unsigned int ei32
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi2 = faceConnections[3*fi3+ei32];
        if ( fi2 != ~0 )
        {
          unsigned int ei23 = findEdge( fi2, fi3, 3, faceConnections );
          DP_ASSERT( ei23 != ~0 );
          ok = checkTriPatch4Start2i( indices, fi2, (ei23+1)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start3( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start3i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start4i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi4, unsigned int ei43
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi3 = faceConnections[3*fi4+ei43];
        if ( fi3 != ~0 )
        {
          unsigned int ei34 = findEdge( fi3, fi4, 3, faceConnections );
          DP_ASSERT( ei34 != ~0 );
          ok = checkTriPatch4Start3i( indices, fi3, (ei34+2)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start4( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start4i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start5i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi5, unsigned int ei51
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi1 = faceConnections[3*fi5+ei51];
        if ( fi1 != ~0 )
        {
          unsigned int ei15 = findEdge( fi1, fi5, 3, faceConnections );
          DP_ASSERT( ei15 != ~0 );
          ok = checkTriPatch4Start1i( indices, fi1, (ei15+1)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start5( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start5i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start6i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi6, unsigned int ei65
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi5 = faceConnections[3*fi6+ei65];
        if ( fi5 != ~0 )
        {
          unsigned int ei56 = findEdge( fi5, fi6, 3, faceConnections );
          DP_ASSERT( ei56 != ~0 );
          ok = checkTriPatch4Start5i( indices, fi5, (ei56+2)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start6( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start6i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start7i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi7, unsigned int ei76
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi6 = faceConnections[3*fi7+ei76];
        if ( fi6 != ~0 )
        {
          unsigned int ei67 = findEdge( fi6, fi7, 3, faceConnections );
          DP_ASSERT( ei67 != ~0 );
          ok = checkTriPatch4Start6i( indices, fi6, (ei67+2)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start7( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start7i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start8i( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi8, unsigned int ei86
                                , const std::vector<unsigned int> & faceConnections
                                , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        unsigned int fi6 = faceConnections[3*fi8+ei86];
        if ( fi6 != ~0 )
        {
          unsigned int ei68 = findEdge( fi6, fi8, 3, faceConnections );
          DP_ASSERT( ei68 != ~0 );
          ok = checkTriPatch4Start6i( indices, fi6, (ei68+1)%3, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool checkTriPatch4Start8( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                               , const std::vector<unsigned int> & faceConnections
                               , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        bool ok = false;
        for ( int i=0 ; i<3 && !ok ; i++ )
        {
          ok = checkTriPatch4Start8i( indices, fi, i, faceConnections, patchIndices, patchFaces );
        }
        return( ok );
      }

      bool FaceConnections::findTriPatch4( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                         , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] )
      {
        DP_ASSERT( m_faceSets.size() == 4 );
        return(   checkTriPatch4Start0( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start1( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start2( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start3( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start4( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start5( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start6( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start7( indices, fi, m_faceConnections, patchIndices, patchFaces )
              ||  checkTriPatch4Start8( indices, fi, m_faceConnections, patchIndices, patchFaces ) );
      }

      unsigned int FaceConnections::getAndClearZeroConnectionIndices( dp::sg::core::IndexSet::ConstIterator<unsigned int> & allIndices
                                                                    , std::vector<unsigned int> & zeroIndices )
      {
        // get the zero-list faces and clear the zero-list
        unsigned int primitiveSize = dp::checked_cast<unsigned int>(m_faceSets.size() - 1);
        for ( std::set<unsigned int>::const_iterator fsit = m_faceSets[0].begin() ; fsit!= m_faceSets[0].end() ; ++fsit )
        {
          unsigned int bi = primitiveSize * *fsit;
          for ( unsigned int i=0 ; i<primitiveSize ; i++ )
          {
            zeroIndices.push_back( allIndices[bi+i] );
          }
        }
        unsigned int ret = dp::checked_cast<unsigned int>(m_faceSets[0].size());
        m_faceSets[0].clear();
        return( ret );
      }

      void FaceConnections::getNeighbours( unsigned int fi, std::vector<unsigned int> & faces )
      {
        unsigned int novpf = dp::checked_cast<unsigned int>(m_faceSets.size() - 1);
        faces.assign( m_faceConnections.begin() + novpf * fi,
                      m_faceConnections.begin() + novpf * fi + novpf );
      }

      unsigned int FaceConnections::getNextFaceIndex( unsigned int * connectivity )
      {
        //  determine the next face index to handle
        for ( unsigned int i=1 ; i<m_faceSets.size() ; i++ )
        {
          if ( ! m_faceSets[i].empty() )
          {
            if ( connectivity )
            {
              *connectivity = i;
            }
            return( *m_faceSets[i].begin() );
          }
        }
        return( ~0 );
      }


      void FaceConnections::connectFaces( dp::sg::core::IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                        , unsigned int ps, unsigned int i0, unsigned int i1
                                        , const std::vector<std::list<unsigned int> > & verticesToFaceLists )
      {
        if ( m_faceConnections[ps*fi+i0] == ~0 )
        {
          unsigned int vi0 = indices[ps*fi+i0];
          unsigned int vi1 = indices[ps*fi+i1];
          //  all faces that share vi0 (except the face we're currently looking at) are candidates for neighbors on edge 01
          const std::list<unsigned int>  &vi0List = verticesToFaceLists[vi0];
          bool found = false;
          for ( std::list<unsigned int>::const_iterator it = vi0List.begin() ; !found && it != vi0List.end() ; ++it )
          {
            if ( *it != fi )
            {
              unsigned int bi = ps * *it;
              for ( unsigned int i=0 ; i<ps && !found ; i++ )
              {
                if ( ( m_faceConnections[bi+i] == ~0 ) && ( indices[bi+i] == vi1 ) && ( indices[bi+(i+1)%ps] == vi0 ) )
                {
                  m_faceConnections[bi+i] = fi;
                  found = true;
                }
              }
              if ( found )
              {
                m_faceConnections[ps*fi+i0] = *it;
              }
            }
          }
        }
      }

    } // namespace algorithm
  } // namespace sg
} // namespace dp
