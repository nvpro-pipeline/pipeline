// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/algorithm/Traverser.h>

namespace dp
{
  namespace sg
  {
    namespace algorithm
    {

      // VertexCacheOptimizer from http://code.google.com/p/vcacne/

      static const int CacheSize = 32;

      class VertexCacheData
      {
      public:
        int position_in_cache;
        float current_score;
        int total_valence; // toatl number of triangles using this vertex
        int remaining_valence; // number of triangles using it but not yet rendered
        std::vector<int> tri_indices; // indices to the indices that use this vertex
        bool calculated; // was the score calculated during this iteration?


        int FindTriangle(int tri)
        {
          for (int i=0; i<(int)tri_indices.size(); i++)
          {
            if (tri_indices[i] == tri) return i;
          }

          return -1;
        }

        void MoveTriangleToEnd(int tri)
        {
          int t_ind = FindTriangle(tri);

          DP_ASSERT(t_ind >= 0);

          tri_indices.erase(tri_indices.begin() + t_ind,
            tri_indices.begin() + t_ind + 1);

          tri_indices.push_back(tri);
        }

        VertexCacheData()
        {
          position_in_cache = -1;
          current_score = 0.0f;
          total_valence = 0;
          remaining_valence = 0;
        }
      };

      class TriangleCacheData
      {
      public:
        bool rendered; // has the triangle been added to the draw list yet?
        float current_score; // sum of the score of its vertices
        int verts[3]; // indices to the triangle's vertices
        bool calculated; // was the score calculated during this iteration?

        TriangleCacheData()
        {
          rendered = false;
          current_score = 0.0f;
          verts[0] = verts[1] = verts[2] = -1;
          calculated = false;
        }
      };

      class VertexCacheOptimizer
      {
      public:
        // CalculateVertexScore constants
        float CacheDecayPower;
        float LastTriScore;
        float ValenceBoostScale;
        float ValenceBoostPower;

        enum class Result
        {
          Success = 0,
          Fail_BadIndex,
          Fail_NoVerts
        };

        static bool Failed(Result r)
        {
          return r != Result::Success;
        }

      protected:
        std::vector<VertexCacheData> verts;
        std::vector<TriangleCacheData> tris;
        std::vector<int> inds;
        std::list<int> vertex_cache;
        std::vector<int> draw_list;

        typedef std::map<float,std::set<int>,std::greater<float> > ScoreMap;
        ScoreMap  triScoreMap;

        float CalculateVertexScore(int vertex)
        {
          VertexCacheData *v = &verts[vertex];
          if (v->remaining_valence <= 0)
          {
            // No tri needs this vertex!
            return -1.0f;
          }

          float ret = 0.0f;
          if (v->position_in_cache < 0)
          {
            // Vertex is not in FIFO cache - no score.
          }
          else
          {
            if (v->position_in_cache < 3)
            {
              // This vertex was used in the last triangle,
              // so it has a fixed score, whichever of the three
              // it's in. Otherwise, you can get very different
              // answers depending on whether you add
              // the triangle 1,2,3 or 3,1,2 - which is silly.
              ret = LastTriScore;
            }
            else
            {
              // Points for being high in the cache.
              const float Scaler = 1.0f / (CacheSize  - 3);
              ret = 1.0f - (v->position_in_cache - 3) * Scaler;
              ret = powf(ret, CacheDecayPower);
            }
          }

          // Bonus points for having a low number of tris still to
          // use the vert, so we get rid of lone verts quickly.
          float valence_boost = powf((float)v->remaining_valence, -ValenceBoostPower);
          ret += ValenceBoostScale * valence_boost;

          return ret;
        }

        float CalculateTriangleScore( int triangle )
        {
          return( verts[tris[triangle].verts[0]].current_score
                + verts[tris[triangle].verts[1]].current_score
                + verts[tris[triangle].verts[2]].current_score );
        }

        void RemoveTriangle( int triangle )
        {
          ScoreMap::iterator it = triScoreMap.find( tris[triangle].current_score );
          DP_ASSERT( it != triScoreMap.end() );
          DP_ASSERT( it->second.find( triangle ) != it->second.end() );
          it->second.erase( triangle );
          if ( it->second.empty() )
          {
            triScoreMap.erase( it );
          }
        }

        void AdjustTriangleScores( int vertex )
        {
          // adjust the score of the triangles using that vertex
          for ( int i=0 ; i<verts[vertex].remaining_valence ; i++ )
          {
            int ti = verts[vertex].tri_indices[i];
            DP_ASSERT( ( tris[ti].verts[0] == vertex ) || ( tris[ti].verts[1] == vertex ) || ( tris[ti].verts[2] == vertex ) );
            float score = CalculateTriangleScore( ti );
            if ( score != tris[ti].current_score )
            {
              RemoveTriangle( ti );
              tris[ti].current_score = score;
              triScoreMap[tris[ti].current_score].insert( ti );
            }
          }
        }

        Result InitialPass()
        {
          for (int i=0; i<(int)inds.size(); i++)
          {
            int index = inds[i];
            if (index < 0 || index >= (int)verts.size())
            {
              return Result::Fail_BadIndex;
            }

            verts[index].total_valence++;
            verts[index].remaining_valence++;

            verts[index].tri_indices.push_back(i/3);
          }

          for (int i=0; i<(int)verts.size(); i++)
          {
            verts[i].current_score = CalculateVertexScore(i);
          }
          for ( int i=0 ; i<(int)tris.size() ; i++ )
          {
            tris[i].current_score = CalculateTriangleScore( i );
            triScoreMap[tris[i].current_score].insert( i );
          }

          return Result::Success;
        }

        Result Init(int *inds, int tri_count, int vertex_count)
        {
          // clear the draw list
          draw_list.clear();

          // allocate and initialize vertices and triangles
          verts.clear();
          for (int i=0; i<vertex_count; i++)
          {
            verts.push_back(VertexCacheData());
          }

          tris.clear();
          for (int i=0; i<tri_count; i++)
          {
            TriangleCacheData dat;
            for (int j=0; j<3; j++)
            {
              dat.verts[j] = inds[i * 3 + j];
            }
            tris.push_back(dat);
          }

          // copy the indices
          this->inds.clear();
          for (int i=0; i<tri_count * 3; i++)
          {
            this->inds.push_back(inds[i]);
          }

          vertex_cache.clear();

          return InitialPass();
        }

        void AddTriangleToDrawList(int tri)
        {
          // reset all cache positions
          for ( std::list<int>::const_iterator it = vertex_cache.begin() ; it != vertex_cache.end() ; ++it )
          {
            verts[*it].position_in_cache = -1;
          }

          TriangleCacheData *t = &tris[tri];
          if (t->rendered)
          {
            return; // triangle is already in the draw list
          }

          for (int i=0; i<3; i++)
          {
            // add all triangle vertices to the cache
            std::list<int>::iterator it = std::find( vertex_cache.begin(), vertex_cache.end(), t->verts[i] );
            if ( it != vertex_cache.end() )
            {
              vertex_cache.erase( it );
            }
            vertex_cache.push_front( t->verts[i] );
            if ( CacheSize == vertex_cache.size() )
            {
              verts[vertex_cache.back()].current_score = CalculateVertexScore( vertex_cache.back() );
              AdjustTriangleScores( vertex_cache.back() );
              vertex_cache.pop_back();
            }

            VertexCacheData *v = &verts[t->verts[i]];

            // decrease remaining velence
            v->remaining_valence--;

            // move the added triangle to the end of the vertex's
            // triangle index list, so that the first 'remaining_valence'
            // triangles in the list are the active ones
            v->MoveTriangleToEnd(tri);
          }

          draw_list.push_back(tri);
          RemoveTriangle( tri );

          t->rendered = true;

          // update all vertex cache positions
          int i = 0;
          for ( std::list<int>::const_iterator it = vertex_cache.begin() ; it != vertex_cache.end() ; ++it, ++i )
          {
            verts[*it].position_in_cache = i;
            verts[*it].current_score = CalculateVertexScore( *it );
            AdjustTriangleScores( *it );
          }
        }

        // Optimization: to avoid duplicate calculations during the same iteration,
        // both vertices and triangles have a 'calculated' flag. This flag
        // must be cleared at the beginning of the iteration to all *active* triangles
        // that have one or more of their vertices currently cached, and all their
        // other vertices.
        // If there aren't any active triangles in the cache, the function returns
        // false and full recalculation is performed.
        bool CleanCalculationFlags()
        {
          bool ret = false;
          for ( std::list<int>::const_iterator it = vertex_cache.begin() ; it != vertex_cache.end() ; ++it )
          {
            VertexCacheData *v = &verts[*it];

            for (int j=0; j<v->remaining_valence; j++)
            {
              TriangleCacheData *t = &tris[v->tri_indices[j]];

              // we actually found a triangle to process
              ret = true;

              // clear triangle flag
              t->calculated = false;

              // clear vertex flags
              for (int tri_vert=0; tri_vert<3; tri_vert++)
              {
                verts[t->verts[tri_vert]].calculated = false;
              }
            }
          }

          return ret;
        }

        int PartialScoreRecalculation()
        {
          // iterate through all the vertices of the cache
          bool first_time = true;
          float max_score;
          int max_score_tri = -1;
          for ( std::list<int>::const_iterator it = vertex_cache.begin() ; it != vertex_cache.end() ; ++it )
          {
            VertexCacheData *v = &verts[*it];

            // iterate through all *active* triangles of this vertex
            for (int j=0; j<v->remaining_valence; j++)
            {
              int tri = v->tri_indices[j];
              TriangleCacheData *t = &tris[tri];
              DP_ASSERT( t->current_score == CalculateTriangleScore( tri ) );
              float sc = t->current_score;

              // we actually found a triangle to process
              if (first_time || sc > max_score)
              {
                first_time = false;
                max_score = sc;
                max_score_tri = tri;
              }
            }
          }

          return max_score_tri;
        }

        // returns true while there are more steps to take
        // false when optimization is complete
        bool Iterate()
        {
          DP_ASSERT( draw_list.size() < tris.size() );

          // recalculate vertex and triangle scores and
          // select the best triangle for the next iteration
          int best_tri;
          if (CleanCalculationFlags())
          {
            best_tri = PartialScoreRecalculation();
          }
          else
          {
            DP_ASSERT( !triScoreMap.empty() );
            best_tri = *(triScoreMap.begin()->second.begin());
          }

          // add the selected triangle to the draw list
          AddTriangleToDrawList(best_tri);

          return( draw_list.size() < tris.size() );
        }

      public:
        VertexCacheOptimizer()
        {
          // initialize constants
          CacheDecayPower = 1.5f;
          LastTriScore = 0.75f;
          ValenceBoostScale = 2.0f;
          ValenceBoostPower = 0.5f;
        }

        // stores new indices in place
        Result Optimize(int *inds, int tri_count)
        {
          // find vertex count
          int max_vert = -1;
          for (int i=0; i<tri_count * 3; i++)
          {
            if (inds[i] > max_vert) max_vert = inds[i];
          }

          if (max_vert == -1) return Result::Fail_NoVerts;

          Result res = Init(inds, tri_count, max_vert + 1);
          if (res != Result::Success) return res;

          // iterate until Iterate returns false
          while (Iterate());

          // rewrite optimized index list
          for (int i=0; i<(int)draw_list.size(); i++)
          {
            inds[3 * i + 0] = tris[draw_list[i]].verts[0];
            inds[3 * i + 1] = tris[draw_list[i]].verts[1];
            inds[3 * i + 2] = tris[draw_list[i]].verts[2];
          }

          return Result::Success;
        }
      };

      /*! \brief Traverse that optimizes the indices and vertices of Primitives of type
        *  PrimitiveType::TRIANGLES by reordering */
      class VertexCacheOptimizeTraverser : public ExclusiveTraverser
      {
        public:
          //! Constructor
          DP_SG_ALGORITHM_API VertexCacheOptimizeTraverser( void );

          //! Destructor
          DP_SG_ALGORITHM_API virtual ~VertexCacheOptimizeTraverser( void );

        protected:
          //! Cleanup temporary memory.
          DP_SG_ALGORITHM_API virtual void postApply( const dp::sg::core::NodeSharedPtr & root );

          //! Optimize the indices of Primitives of type PrimitiveType::TRIANGLES
          DP_SG_ALGORITHM_API virtual void handlePrimitive( dp::sg::core::Primitive * p );

        private:
          VertexCacheOptimizer    m_vco;
          std::vector<int>        m_newIndices;
          std::set<const void *>  m_objects;      //!< A set of pointers to hold all objects already encountered.
      };

    } // namespace algorithm
  } // namespace sg
} // namespace dp
