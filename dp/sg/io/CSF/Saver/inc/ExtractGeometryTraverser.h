// Copyright NVIDIA Corporation 2002-2012
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
/** \file */

#include <dp/math/Trafo.h>
#include <dp/sg/algorithm/Traverser.h>
#include <dp/sg/algorithm/TransformStack.h>
#include <dp/sg/core/EffectData.h>
#include <dp/sg/core/Primitive.h>
#include <dp/sg/io/CSF/Saver/inc/CSFSGWrapper.h>

#include <string>


// Traverser that extracts the geometry of the scene.
class ExtractGeometryTraverser : public dp::sg::algorithm::SharedTraverser
{
  public:
    /** \note A ExtractGeometryTraverser doesn't change anything in the scene graph, but might be used by a modifying traverser.
      * therefore, it gets the readOnly flag as a parameter to pass to Traverser. */
    ExtractGeometryTraverser();

    virtual ~ExtractGeometryTraverser();

    std::vector<CSFSGNode>& getNodes();
    std::vector<CSFSGMaterial>& getMaterials();
    std::vector<CSFSGGeometry>& getGeometries();

    typedef std::map <const dp::sg::core::EffectData*,int>  CSFSGMaterialHashMap;
    typedef std::pair<const dp::sg::core::EffectData*,int>  CSFSGMaterialHashPair;

    typedef std::pair<int,int>                                                CSFSGGeometryHashEntry;
    typedef std::pair<const dp::sg::core::Primitive*,CSFSGGeometryHashEntry>  CSFSGGeometryHashPair;
    typedef std::map <const dp::sg::core::Primitive*,CSFSGGeometryHashEntry>  CSFSGGeometryHashMap;

  protected:

    // Provide special treatment of a LOD node.
    virtual void handleLOD( const dp::sg::core::LOD *lod );

    // Provide special treatment of a Geonode.
    virtual void handleGeoNode( const dp::sg::core::GeoNode * p );

    //! Provide special treatment of a Transform node.
    virtual void handleTransform( const dp::sg::core::Transform *p );

    // Handles actions to take between transform stack adjustment and traversal.
    virtual bool preTraverseTransform( const dp::math::Trafo *p );

    //! Handles actions to take between traversal and transform stack adjustment.
    virtual void postTraverseTransform( const dp::math::Trafo *p );

    virtual void traversePrimitive( const dp::sg::core::Primitive * p );


  private:

    int makeIDX (size_t idx)
    {
      return dp::util::checked_cast<int,size_t>(idx);
    }

    int addPrimitive(int geometryIDX, const dp::sg::core::Primitive* primitive);


    std::vector<CSFSGNode>      m_nodes;
    std::vector<CSFSGMaterial>  m_materials;
    std::vector<CSFSGGeometry>  m_geometries;

    CSFSGMaterialHashMap m_materialMap;
    CSFSGGeometryHashMap m_geometryMap;
    
    int  m_materialIDX;
    int  m_objectIDX;


    std::stack<int> m_parentStack;
    std::stack<int> m_objectStack;


    std::string   m_annotation;
    
};
