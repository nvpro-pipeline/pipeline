// Copyright NVIDIA Corporation 2002-2009
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

#include <dp/sg/core/Config.h>
#include <dp/sg/core/IndexSet.h>
#include <set>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Helper class to determine face connectivity from triangle or quad soups.
       *  \remarks This class is used, for example, by the BezierInterpolationTraverser, and the
       *  StrippingTraverser.
       *  \sa BezierInterpolationTraverser, StrippingTraverser */
      class FaceConnections
      {
        public:
          /*! \brief Constructor of a FaceConnections object.
           *  \param p A pointer to the Primitive to handle.
           *  \remarks All the connectivity informations about the Primitive \a p is determined.
           *  Depending on the size of \a p, that might take a while.
           *  \sa Primitive */
          DP_SG_CORE_API FaceConnections( const Primitive * p );

          /*! \brief Disconnect a single face from the connectivity set.
           *  \param faceIndex The index of the face to disconnect.
           *  \remarks The connections from all faces adjacent to this face are removed, reducing the connections
           *  count of those faces accordingly. Essentially, the face \a faceIndex is removed. */
          DP_SG_CORE_API void disconnectFace( unsigned int faceIndex );

          /*! \brief Disconnect an array of faces from the connectivity set.
           *  \param faceIndices A pointer to face indices to disconnect.
           *  \param faceCount The number of faces to disconnect.
           *  \remarks All faces from \a faceIndices[0] to \a faceIndices[\a faceCount-1] are disconnected.
           *  \sa disconnectFace */
          DP_SG_CORE_API void disconnectFaces( const unsigned int * faceIndices, unsigned int faceCount );

          /*! \brief Disconnect a list of faces from the connectivity set.
           *  \param list A reference to a list of face indices to disconnect.
           *  \remarks All faces in the list \a list are disconnected.
           *  \sa disconnectFace */
          DP_SG_CORE_API void disconnectFaces( const std::list<unsigned int> &list );

          /*! \brief Find the longest quad strip in the quad soup and append the respective vertex indices to \a stripIndices.
           *  \param indices A pointer to the indices of the quads.
           *  \param fi Index of the face to start the quad strip determination at.
           *  \param stripIndices A reference to a vector getting the vertex indices of the determined quad strip.
           *  \param stripFaces A reference to a list getting the face indices of the determined quad strip.
           *  \return Length of the longest quad strip found.
           *  \remarks Determines the 'horizontal' and the 'vertical' quad strip including the face \a fi and selects
           *  the longer one.
           *  \sa findLongestTriStrip */
          DP_SG_CORE_API unsigned int findLongestQuadStrip( IndexSet::ConstIterator<unsigned int> &, unsigned int fi
                                                          , std::vector<unsigned int> & stripIndices
                                                          , std::list<unsigned int> & stripFaces );

          /*! \brief Find the longest tri strip in the triangle soup and append the respective vertex indices to \a stripIndices.
           *  \param indices A pointer to the indices of the triangles.
           *  \param fi Index of the face to start the tri strip determination at.
           *  \param stripIndices A reference to a vector getting the vertex indices of the determined tri strip.
           *  \param stripFaces A reference to a list getting the face indices of the determined tri strip.
           *  \return Length of the longest tri strip found.
           *  \remarks Determines the three possible tri strip including the face \a fi, and selects the longest one.
           *  \fineLongestQuadStrip */
          DP_SG_CORE_API unsigned int findLongestTriStrip( IndexSet::ConstIterator<unsigned int> & indices
                                                         , unsigned int fi, std::vector<unsigned int> & stripIndices
                                                         , std::list<unsigned int> & stripFaces );

          /*! \brief Determine if a quad can be part of a connected set of 3 x 3 quads.
           *  \param indices A pointer to the indices of the quads.
           *  \param fi Index of the face to use for the test.
           *  \param patchIndices A reference to a vector getting the vertex indices of the determined set of 3 x 3 quads.
           *  \param patchFaces An array to get the 9 face indices forming the QuadPatch4x4.
           *  \return \c true if a set of quads was found, otherwise \c false.
           *  \remarks A QuadPatch4x4 consists of a set of 3 x 3 quads, or 4 x 4 vertices. This function simply checks
           *  the connectivities of the face \a fi and its neighbours, whether it forms such a set of quads.
           *  \sa findTriPatch4 */
          DP_SG_CORE_API bool findQuadPatch4x4( IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                        , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] );

          /*! \brief Determine if a triangle can be part of a triangular connected set of 9 triangles.
           *  \param indices A pointer to the indices of the triangles.
           *  \param fi Index of the face to use for the test.
           *  \param patchIndices A reference to a vector getting the vertex indices of the determined set of triangles.
           *  \param patchFaces An array to get the 9 face indices forming the TriPatch4.
           *  \return \c true if a set of triangles was found, otherwise \c false.
           *  \remarks A TriPatch4 consists of a set of 9 triangles, that are arranged to form a triangular shape again.
           *  This function simply checks the connectivities of the face \a fi and it neighours, whether it forms such a
           *  set of triangles.
           *  \sa findQuadPatch4x4 */
          DP_SG_CORE_API bool findTriPatch4( IndexSet::ConstIterator<unsigned int> & indices, unsigned int fi
                                           , std::vector<unsigned int> & patchIndices, unsigned int patchFaces[9] );

          /*! \brief Get all the faces without any neighbours, and clear the list that holds them
           *  \param allIndices A pointer to the indices of the primitive set.
           *  \param zeroIndices A reference to a vector getting the vertex indices of the isolated primitives.
           *  \return The number of faces without any neighbours that were in the connectivity set.
           *  \remarks After having determined all strips or patches out of the primitive set, there still might be some
           *  primitives that had no neighbours, or were isolated in the stripping/patching process. You get all the
           *  vertex indices of those faces with this function.
           *  \sa findLongestQuadStrip, findLongestTriStrip, findQuadPatch4x4, findTriPatch4 */
          DP_SG_CORE_API unsigned int getAndClearZeroConnectionIndices( IndexSet::ConstIterator<unsigned int> & allIndices
                                                                      , std::vector<unsigned int> & zeroIndices );

          /*! \brief Get the neighbours of a primitive.
           *  \param fi Index of the face to get the neighbours of.
           *  \param faces A reference to a vector getting the face indices of the neighbours of \a fi.
           *  \sa getNextFaceIndex */
          DP_SG_CORE_API void getNeighbours( unsigned int fi, std::vector<unsigned int> & faces );

          /*! \brief Get the next face index with at least one neighbour.
           *  \param connectivity Optional parameter to get the connectivity of the next face index.
           *  \return The index of the next face to handle, or ~0 if there is none.
           *  \sa getNeighbours */
          DP_SG_CORE_API unsigned int getNextFaceIndex( unsigned int * connectivity = NULL );

        private:
          void connectFaces( IndexSet::ConstIterator<unsigned int> & indices, unsigned int faceIndex
                           , unsigned int verticesPerFace, unsigned int edgeIndex0, unsigned int edgeIndex1
                           , const std::vector<std::list<unsigned int> > & verticesToFaceLists );

        private:
          std::vector<unsigned int>             m_faceConnections;
          std::vector<unsigned int>             m_faceConnectionCounts;
          std::vector<std::set<unsigned int> >  m_faceSets;
      };

    } // namespace core
  } // namespace sg
} // namespace dp


