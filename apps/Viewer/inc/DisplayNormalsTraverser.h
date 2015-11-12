// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/sg/algorithm/Traverser.h>

#define NORMALS_NAME "__displayNormals__"

//! Traverser that can add visual representations of the scene's vertex normals.
class DisplayNormalsTraverser : public dp::sg::algorithm::ExclusiveTraverser
{
  public:
    //! Constructor
    DisplayNormalsTraverser();

    //! Destructor
    virtual ~DisplayNormalsTraverser(void);

    //! Set the length of the vertex normals to display
    void setNormalLength(float len);

    //! Set the color of the vertex normals to display
    void setNormalColor(dp::math::Vec3f &color);

  protected:
    /*! \brief Handle the special case for the root being a GeoNode.
     *  \param root The root of the tree to handle.
     *  \remarks The behaviour is undefined, if \a root is not the root node of the currently handled scene. */
    virtual void doApply( const dp::sg::core::NodeSharedPtr & root );

    /*! \brief Handle a GeoNode.
     *  \param p A pointer to the GeoNode to handle.
     *  \remarks A corresponding GeoNode with lines is created, representing the normals of the Primitive of \a p. */
    virtual void handleGeoNode( dp::sg::core::GeoNode *p );

    //! Handle a Primitive
    /** The Primitive's may contain old normals which we want to remove.  */
    virtual void handlePrimitive( dp::sg::core::Primitive *p );

    //! Handle a Transform object.
    /** We need to store the transform's scale to scale the normals. */
    virtual void handleTransform( dp::sg::core::Transform *p );

    /*! \brief Handle a Group object.
     *  \param p The Group to handle.
     *  \remarks Out of each GeoNode directly beneath this Group, a new GeoNode holding the normals (as lines)
     *  is created. A Group holding the original GeoNode and that new GeoNode replaces the original GeoNode in
     *  this Group. */
    virtual void handleGroup( dp::sg::core::Group * p );

    /*! \brief Handle a Switch object.
     *  \param p The Switch to handle.
     *  \remarks Out of each GeoNode directly beneath this Switch, a new GeoNode holding the normals (as lines)
     *  is created. A Group holding the original GeoNode and that new GeoNode replaces the original GeoNode in
     *  this Switch. */
    virtual void handleSwitch( dp::sg::core::Switch * p );

    /*! \brief Handle a Switch object.
     *  \param p The LOD to handle.
     *  \remarks Out of each GeoNode directly beneath this LOD, a new GeoNode holding the normals (as lines)
     *  is created. A Group holding the original GeoNode and that new GeoNode replaces the original GeoNode in
     *  this LOD. */
    virtual void handleLOD( dp::sg::core::LOD * p );

  private:
    void checkNormals( dp::sg::core::Group * p );

  private:
    std::map<dp::sg::core::VertexAttributeSetSharedPtr,std::set<unsigned int> >                         m_indices;
    dp::sg::core::PipelineDataSharedPtr                                                                 m_material;
    float                                                                                               m_normalLength;
    std::stack<std::vector<std::pair<dp::sg::core::GeoNodeSharedPtr,dp::sg::core::GeoNodeSharedPtr> > > m_normalsGeoNodes;
    std::stack<std::vector<dp::sg::core::GroupSharedPtr> >                                              m_normalsGroups;
    std::vector<std::pair<dp::sg::core::PrimitiveSharedPtr,dp::sg::core::PrimitiveSharedPtr> >          m_normalsPrimitives;
    dp::math::Trafo                                                                                     m_trafo;
};

inline void DisplayNormalsTraverser::setNormalLength(float len)
{
  m_normalLength = len;
}
