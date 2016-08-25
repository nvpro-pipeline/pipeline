// Copyright NVIDIA Corporation 2002-2015
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
#include <dp/sg/core/BoundingVolumeObject.h>
#include <dp/sg/core/IndexSet.h>
#include <dp/sg/core/VertexAttributeSet.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief PrimitiveTypes associated with this Primitive */
      enum class PrimitiveType
      {
        // default Primitive types
        POINTS,
        LINE_STRIP,
        LINE_LOOP,
        LINES,
        TRIANGLE_STRIP,
        TRIANGLE_FAN,
        TRIANGLES,
        QUAD_STRIP,
        QUADS,
        POLYGON,

        //  If NV_geometry_program4 is supported, then the following primitive types are also supported:
        TRIANGLES_ADJACENCY,
        TRIANGLE_STRIP_ADJACENCY,
        LINES_ADJACENCY,
        LINE_STRIP_ADJACENCY,

        // if NV_tessellation_program5 / ARB_tessellation_shader is supported
        PATCHES,

        UNINITIALIZED = ~0
      };

      DP_SG_CORE_API PrimitiveType primitiveNameToType( std::string const& name );
      DP_SG_CORE_API std::string primitiveTypeToName( PrimitiveType pt );

      enum class PatchesMode
      {
        TRIANGLES,
        QUADS,
        ISOLINES,
        POINTS
      };

      enum class PatchesSpacing
      {
        EQUAL,
        FRACTIONAL_EVEN,
        FRACTIONAL_ODD
      };

      enum class PatchesOrdering
      {
        CW,
        CCW
      };

      enum class PatchesType
      {
        NONE,
        PN_TRIANGLES,
        PN_QUADS,
        CUBIC_BEZIER_TRIANGLES,
        CUBIC_BEZIER_QUADS
      };

      DP_SG_CORE_API PatchesType patchesNameToType( std::string const& name );
      DP_SG_CORE_API std::string patchesTypeToName( PatchesType pt );
      DP_SG_CORE_API unsigned int verticesPerPatch( PatchesType pt );


      /*! \brief Class for Primitive objects.
       *  \sa VertexAttributeSet, IndexSet */
      class Primitive : public BoundingVolumeObject
      {
        public:
          DP_SG_CORE_API static PrimitiveSharedPtr create( PrimitiveType pt );
          DP_SG_CORE_API static PrimitiveSharedPtr create( PatchesType pt, PatchesMode pm = PatchesMode::TRIANGLES );

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;
          DP_SG_CORE_API PrimitiveSharedPtr cloneAs(PrimitiveType pt);

          DP_SG_CORE_API virtual ~Primitive();

        public:
          // internal render flags used to know how to draw this primitive
          enum
          {
            DRAW_ARRAYS     = 0,      // here to make code more readable
            DRAW_INDEXED    = BIT(0),
            DRAW_INSTANCED  = BIT(1),
          };

          /*! \brief Assignment operator
           *  \param rhs A reference to the constant Primitive to copy from.
           *  \return A reference to the assigned Primitive.
           */
          DP_SG_CORE_API Primitive & operator=( const Primitive & rhs );

          /*! \brief Get the primitive type
           *  \param pt The primitive type to use to render this primitive.
           *  \remarks If the selected primitive type is not available on the target hardware, nothing will be drawn for this primitive.
           *  \sa PrimitiveType */
          DP_SG_CORE_API PrimitiveType getPrimitiveType() const;

          DP_SG_CORE_API PatchesType getPatchesType() const;
          DP_SG_CORE_API PatchesMode getPatchesMode() const;

          DP_SG_CORE_API void setPatchesSpacing( PatchesSpacing ps );
          DP_SG_CORE_API PatchesSpacing getPatchesSpacing() const;

          DP_SG_CORE_API void setPatchesOrdering( PatchesOrdering po );
          DP_SG_CORE_API PatchesOrdering getPatchesOrdering() const;

          /*! \brief Set the range of elements to use.
           * If an IndexSet is present the range applies to that, if not, it applies to the VertexAttributeSet directly.
           * \param offset The starting index or vertex to use for this primitive. 
           * The default value is 0, meaning the traverser should begin sourcing indices or vertices from the start 
           * of the IndexSet or VertexAttributeSet.
           * \param count The count of indices or vertices to use for this primitive.
           * The default value of ~0 describes the end of the IndexSet or VertexAttributeSet. 
           * It will cause all elements from offset to the end of the IndexSet or VertexAttributeSet to be used.
           * Results are undefined if offset is greater than the size of the IndexSet or VertexArray.
           * Results are undefined if the count is not equal to ~0 and offset + count is greater than the IndexSet or VertexAttributeSet size.
           * \remarks Sets the number of elements to use for this primitive. This is the number of indices or vertices, NOT the
           * number of primitives. 
           * \note This method must be called \b after setIndexSet() or setVertexAttributeSet() for proper validation.
           * The defaults match the call setElementRange(0, ~0) which will render the whole Primitive.
           * With these defaults it is not required to update the range after the IndexSet or VertexAttributeSet size has been changed 
           * if the whole Primitive should be rendered.
           * \sa getElementRange, getElementOffset, getElementCount, getMaxElementCount
           */
          DP_SG_CORE_API void setElementRange( unsigned int offset, unsigned int count );

          /*! \brief Get the element range values offset and count exactly as specified by the user via setElementRange.
           * If setElementRange hasn't been used the returned values will be offset = 0 and count = ~0 which describes 
           * the whole IndexSet or VertexAttributeSet.
           * \param offset Returns the user defined offset value for the starting index or vertex.
           * \param count Returns the user defined count value for indices or vertices.
           * This can be ~0 which describes the IndexSet or VertexAttributeSet end.
           * \remarks The range describes the start and count of indices or vertices to use, \b NOT the number of primitives.
           * setElementRange and getElementRange are symmetric to be able to use them in succession while the special count parameter ~0 is used.
           * To get the effective number of elements use getElementCount. It will return the user count if it is not ~0 and
           * the IndexSet or VertexAttributeSet size minus offset when the user count is ~0.
           * \sa setElementRange, getElementOffset, getElementCount, getMaxElementCount
           */
          DP_SG_CORE_API void getElementRange( unsigned int & offset, unsigned int & count ) const;

          /*! \brief Get the user defined element offset value.
           * Returns the user defined offset value for the starting index or vertex.
           * If setElementRange hasn't been used the returned value will be 0 describing the start of the IndexSet or VertexAttributeSet.
           * \remarks The offset value describes the start of indices or vertices to use, \b NOT the number of primitives.
           * \sa setElementRange, getElementRange, getElementCount, getElementCount
           */
          DP_SG_CORE_API unsigned int getElementOffset() const;

          /*! \brief Get the \b effective element count.
           * Returns the effective count value for indices or vertices.
           * \remarks The count describes the number of indices or vertices, \b NOT the number of primitives.
           * With the default element count value of ~0 this function returns the IndexSet or VertexAttribute size minus the element offset.
           * This is in contrast to the getElementRange function which returns the user defined range values which can have count at ~0.
           * There is no setElementCount defined on purpose because that wouldn't be symmetric to this getElementCount function.
           * \sa setElementRange, getElementRange, getElementOffset, getMaxElementCount
           */
          DP_SG_CORE_API unsigned int getElementCount() const;

          /*! \brief Get the maximum allowed element count of the IndexSet or VertexAttributeSet.
           * This is a convenience function which allows to query the IndexSet or VertexAttributeSet size 
           * without considering the currently active user defined element range offset or count values.
           * Only with the default setElementRange(0, ~0) setting the result will match the one from getElementCount.
           * \sa setElementRange, getElementRange, getElementOffset, getElementCount
           */
          DP_SG_CORE_API unsigned int getMaxElementCount() const;

          /*! \brief Set the instance count for this primitive.
           * \param icount The number of times to instance (render) this primitive.  Typically the shader used to render an instanced primitive 
           * must be written to be instance-aware, and separate data must be packed into vertex attributes in order to appropriately render each 
           * instance.  The value must be greater than or equal to 1.
           * \remarks If icount is greater than 1, and OpenGL 3.0 or ARB_draw_instanced is not available, nothing will be rendered for this 
           * primitive since a compatible software fallback is not possible.
           *  \note The default instance count is 1, rendering exactly 1 primitive.
           */
          DP_SG_CORE_API void setInstanceCount( unsigned int icount );
          DP_SG_CORE_API unsigned int getInstanceCount() const;

          /*! \brief Set the VertexAttributeSet.
           *  \param vash The VertexAttributeSet to set.
           *  \sa VertexAttributeSet */
          DP_SG_CORE_API void setVertexAttributeSet( const VertexAttributeSetSharedPtr & vash );

          /*! \brief Get The VertexAttributeSet of this Primitive.
           *  \return The VertexAttributeSet of this Primitive.
           *  \sa VertexAttributeSet */
          DP_SG_CORE_API const VertexAttributeSetSharedPtr & getVertexAttributeSet() const;

          /*! \brief Get the IndexSet 
           *  \return The IndexSet.
           *  \note If a Primitive does not contain an IndexSet then it is considered to be a non-indexed array of primitives 
           *  that will be drawn in linear order as specified in the VAS.
           *  \sa IndexSet */
          DP_SG_CORE_API const IndexSetSharedPtr & getIndexSet() const;

          /*! \brief Set the IndexSet.
           *  \param iset A pointer to the IndexSet.
           *  \sa IndexSet, getIndexSet */
          DP_SG_CORE_API void setIndexSet( const IndexSetSharedPtr & iset );

          /*! \brief Check whether this Primitive is indexed or not.
           *  \return \c true if the Primitive \a p is indexed (contains a IndexSet), otherwise \c false.
           *  \sa IndexSet, setIndexSet */
          DP_SG_CORE_API bool isIndexed() const;

          /*! \brief Make this Primitive indexed.
           *  \return \c true if the Primitive was not indexed before, otherwise \c false.
           *  \remark If this Primitive is not indexed, an IndexSet is created, indexing the vertices
           *  from offset to offset+count. Otherwise nothing is done.
           *  \sa isIndexed, getIndexSet, setIndexSet, getElementOffset, getElementCount */
          DP_SG_CORE_API bool makeIndexed();

          /*! \brief Test for equivalence with an other Primitive.
           *  \param p A reference to the constant Primitive to test for equivalence with.
           *  \param ignoreNames Optional parameter to ignore the names of the objects; default is \c
           *  true.
           *  \param deepCompare Optional parameter to perform a deep comparsion; default is \c false.
           *  \return \c true if the Primitive \a p is equivalent to \c this, otherwise \c false.
           *  \remarks If \a p and \c this are equivalent as Object, they are equivalent if the
           *  VertexAttributeSet objects are equivalent. If \a deepCompare is \c true, a full
           *  equivalence test is performed on the VertexAttributeSet objects, otherwise they are
           *  considered to be equivalent if the pointers are equal.
           *  \note The behavior is undefined if \a p is not a Primitive nor derived from one.
           *  \sa Object, VertexAttributeSet */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          /*! \brief Get the renderer specific render flags.
           *  \remarks This function is used by the renderer to accelerate rendering of this primitive
           *  \return The render flags
           */
          DP_SG_CORE_API unsigned int getRenderFlags() const;

          /*! \brief Get the number of geometric primitives in this Primitive.
           *  \remarks This function returns the number of geometric primitives in this primitive.  This count varies
           *  depending on the primitive type, whether the primitive is indexed, is stripped, and whether the IndexSet
           *  contains Primitive Restart indices or not.  If the primitive is "singular" (*_STRIP, TRIANGLE_FAN, etc) the 
           *  count of that type (not the face count) is returned.  If the primitive is "plural" (TRIANGLES, QUADS, etc) the 
           *  face count is returned.
           *  \return The primitive count.
           *  \sa getNumberOfFaces, getNumberOfVerticesPerPrimitive
           */
          DP_SG_CORE_API unsigned int getNumberOfPrimitives() const;

          /*! \brief Get the number of faces in this primitive.
           *  \remarks This function returns the number of faces in this primitive.  The face count varies
           *  depending on the primitive type, whether the primitive is indexed, is stripped, and whether the IndexSet
           *  contains Primitive Restart indices or not.  If the primitive is "singular" (*_STRIP, TRIANGLE_FAN, etc) the 
           *  primitive will be processed to determine the actual number of faces in the strip or fan, etc.  If the primitive 
           *  is "plural" (TRIANGLES, QUADS, etc) the face count equals the number of primitives.
           *  \return The face count.
           *  \sa getNumberOfVerticesPerPrimitive, getNumberOfPrimitives
           */
          DP_SG_CORE_API unsigned int getNumberOfFaces() const;

          /*! \brief Get the number of vertices per Primitive
           *  \remarks This function returns the number of vertices required to specify a complete Primitive.  For instance
           *  it would be 3 for PrimitiveType::TRIANGLES, 4 for PrimitiveType::QUADS, etc.  For types with a variable number of 
           *  vertices per primitive (PrimitiveType::POLYGON, PrimitiveType::TRIANGLE_STRIP, etc) the return value is zero.
           *  \return The vertex count, 0, or ~0 if the primitive is uninitialized.
           *  \sa getNumberOfFaces, getNumberOfPrimitives, getNumberOfPrimitiveRestarts
           */
          DP_SG_CORE_API unsigned int getNumberOfVerticesPerPrimitive() const;

          /*! \brief Get the number of primitive restarts in the IndexSet.
           *  \remarks This function returns the number of primitive restarts found in the IndexSet.
           *  \return The number of primitive restarts found in the IndexSet, which may be zero if there is no index set or 
           *  no primitive restarts therein.
           *  \sa getNumberofFaces, getNumberOfElements, getNumberOfVerticesPerPrimitive
           */
          DP_SG_CORE_API unsigned int getNumberOfPrimitiveRestarts() const;

          /*! \brief Generates vertex normals 
           *  \param overwrite An optional flag indicating whether to overwrite existing vertex normals.
           *  The default is to overwrite existing data.
           *  \return \c true, if normals could be generated, otherwise \c false.
           *  \remarks The function calls the protected virtual function calculateNormals.
           * \sa calculateNormals */
          DP_SG_CORE_API bool generateNormals( bool overwrite = true );

          /*! \brief Generates tangents and binormals
           * \param texcoords
           * Addresses the vertex attribute to hold the input 2D texture coordinates used to calculate the tangent space.
           * By default, input texture coordinates are taken from the VertexAttributeSet::AttributeID::TEXCOORD0. 
           * \param tangents
           * Addresses the vertex attribute where to output the calculated tangents. 
           * By default tangents are written to VertexAttributeSet::AttributeID::TANGENT, 
           * which is aligned to the TANGENT binding semantic used by Cg for varying vertex shader input. 
           * \param binormals
           * Addresses the vertex attribute where to output the calculated binormals.
           * By default binormals are written to VertexAttributeSet::AttributeID::BINORMAL, 
           * which is aligned to the BINORMAL binding semantic used by Cg for varying vertex shader input. 
           * \param overwrite 
           * An optional flag indicating whether to overwrite existing vertex data.
           * The default is to overwrite existing data.
           * \remarks
           * The function calls the protected virtual function calculateTangentSpace, which concrete Drawables
           * should override to provide correct tangent space calculation.
           * \sa calculateTangentSpace */
          DP_SG_CORE_API void generateTangentSpace( VertexAttributeSet::AttributeID texcoords = VertexAttributeSet::AttributeID::TEXCOORD0, 
                                                    VertexAttributeSet::AttributeID tangents  = VertexAttributeSet::AttributeID::TANGENT, 
                                                    VertexAttributeSet::AttributeID binormals = VertexAttributeSet::AttributeID::BINORMAL,
                                                    bool overwrite = true );

          /*! \brief Generates 2D texture coordinates 
           * \param type
           * Desired texture coordinate type. Accepted are TextureCoordType::CYLINDRICAL, TextureCoordType::PLANAR, and TextureCoordType::SPHERICAL. 
           * \param texcoords 
           * Addresses the vertex attribute where to output the generated texture coords. 
           * VertexAttributeSet::AttributeID::TEXCOORD0 - VertexAttributeSet::AttributeID::TEXCOORD7 are allowed identifiers.
           * By default texture coords are written to VertexAttributeSet::AttributeID::TEXCOORD0, 
           * \param overwrite 
           * An optional flag indicating whether to overwrite existing vertex data.
           * The default is to overwrite existing data.
           * \remarks
           * The function calls the protected virtual function calculateTexCoords, which concrete Drawables
           * should override to provide correct texture coordinate calculation.
           * \sa calculateTexCoords */
          DP_SG_CORE_API void generateTexCoords( TextureCoordType type, 
                                                 VertexAttributeSet::AttributeID texcoords = VertexAttributeSet::AttributeID::TEXCOORD0, 
                                                 bool overwrite = true );

          REFLECTION_INFO_API( DP_SG_CORE_API, Primitive );

        protected:
          DP_SG_CORE_API Primitive( PrimitiveType primitiveType, PatchesType patchesType, PatchesMode patchesMode );

          /*! \brief Copy constructor
           *  \param rhs A reference to the constant Primitive to copy from
           *  \remarks The new Primitive holds a copy of the VertexAttributeSet and IndexSet (if any) of \a rhs. */
          DP_SG_CORE_API Primitive( const Primitive& rhs );

          DP_SG_CORE_API void determinePrimitiveAndFaceCount() const;

          /*! \brief Override to specialize calculation of texture coords */
          DP_SG_CORE_API virtual void calculateTexCoords(TextureCoordType type, VertexAttributeSet::AttributeID texcoords, bool overwrite);

          /*! \brief Calculates the bounding box of this Primitive.
           *  \return The axis-aligned bounding box of this Primitive.
           *  \remarks This function is called by the framework when re-calculation
           *  of the bounding box is required for this Primitive.
           *  Overrides BoundingVolumeObject::calculateBoundingBox. */
          DP_SG_CORE_API virtual dp::math::Box3f calculateBoundingBox() const;

          /*! \brief Calculate the bounding sphere of this Primitive.
           *  \return A dp::math::Sphere3f that contains the complete Primitive.
           *  \remarks This function is called by the framework to determine a sphere that completely
           *  contains the Primitive.
           *  Overrides BoundingVolumeObject::calculateBoundingSphere. */
          DP_SG_CORE_API virtual dp::math::Sphere3f calculateBoundingSphere() const;

          /*! \brief Override to specialize normals calculation for Primitives
           *  \param overwrite A flag indicating whether to overwrite existing vertex normals.
           *  \return \c true, if normals could be calculated, otherwise \c false.
           *  \remarks This function gets called from the generateNormals API. If \a overwrite is \c true or the
           *  Primitive has no normals, and the Primitive is of type PrimitiveType::TRIANGLE_STRIP, PrimitiveType::TRIANGLE_FAN,
           *  PrimitiveType::TRIANGLES, PrimitiveType::QUAD_STRIP, or PrimitiveType::QUADS, normals are calculated and \c true is
           *  returned. Otherwise \c false is returned.
           *  \sa generateNormals */
          DP_SG_CORE_API virtual bool calculateNormals(bool overwrite);

          /*! \brief Override to specialize tangent space calculation for Primitives
           *  \param texcoords Addresses the vertex attribute to hold the input 2D texture coordinates used to calculate
           *  the tangent space.
           *  \param tangents Addresses the vertex attribute where to output the calculated tangents.
           *  \param binormals Addresses the vertex attribute where to output the calculated binormals.
           *  \param overwrite A flag indicating whether to overwrite existing vertex data.
           *  \remarks This function gets called from the generateTangentSpace API. If \a overwrite is \c true or the
           *  Primitive has no normals, and the Primitive is of type PrimitiveType::TRIANGLE_STRIP, PrimitiveType::TRIANGLE_FAN,
           *  PrimitiveType::TRIANGLES, PrimitiveType::QUAD_STRIP, or PrimitiveType::QUADS, tangents and binormals are calculated.
           * \sa generateTangentSpace */
          DP_SG_CORE_API virtual void calculateTangentSpace(VertexAttributeSet::AttributeID texcoords, VertexAttributeSet::AttributeID tangents, VertexAttributeSet::AttributeID binormals, bool overwrite);

          /*! \brief Feed the data of this object into the provied HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          bool calculateNormals( const VertexAttributeSetSharedPtr& vassp, bool overwrite );
          void calculateNormalsPolygon( Buffer::ConstIterator<dp::math::Vec3f>::Type & vertices
                                      , std::vector<dp::math::Vec3f> & normals );
          void calculateNormalsQuad( Buffer::ConstIterator<dp::math::Vec3f>::Type & vertices
                                   , std::vector<dp::math::Vec3f> & normals );
          void calculateNormalsQuadStrip( Buffer::ConstIterator<dp::math::Vec3f>::Type & vertices
                                        , std::vector<dp::math::Vec3f> & normals );
          void calculateNormalsTriangle( Buffer::ConstIterator<dp::math::Vec3f>::Type & vertices
                                       , std::vector<dp::math::Vec3f> & normals );
          void calculateNormalsTriFan( Buffer::ConstIterator<dp::math::Vec3f>::Type & vertices
                                     , std::vector<dp::math::Vec3f> & normals );
          void calculateNormalsTriStrip( Buffer::ConstIterator<dp::math::Vec3f>::Type & vertices
                                       , std::vector<dp::math::Vec3f> & normals );
          void calculateTangentsQuad( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc
                                    , std::vector<dp::math::Vec3f> & tangents );
          void calculateTangentsQuadStrip( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc
                                         , std::vector<dp::math::Vec3f> & tangents );
          void calculateTangentsTriangle( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc
                                        , std::vector<dp::math::Vec3f> & tangents );
          void calculateTangentsTriFan( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc
                                      , std::vector<dp::math::Vec3f> & tangents );
          void calculateTangentsTriStrip( VertexAttributeSetSharedPtr const& vas, VertexAttributeSet::AttributeID tc
                                        , std::vector<dp::math::Vec3f> & tangents );

          void clearCachedCounts() const;

        private:
          PrimitiveType               m_primitiveType;
          PatchesType                 m_patchesType;
          PatchesMode                 m_patchesMode;
          PatchesSpacing              m_patchesSpacing;
          PatchesOrdering             m_patchesOrdering;
          VertexAttributeSetSharedPtr m_vertexAttributeSet;
          IndexSetSharedPtr           m_indexSet;
          unsigned int                m_elementOffset;
          unsigned int                m_elementCount;
          unsigned int                m_instanceCount;
          unsigned int                m_renderFlags;
          mutable dp::math::Box3f     m_boundingBox;
          mutable dp::math::Sphere3f  m_boundingSphere;
          mutable unsigned int        m_cachedNumberOfPrimitives;
          mutable unsigned int        m_cachedNumberOfFaces;
          mutable unsigned int        m_cachedNumberOfPrimitiveRestarts;
      };

      inline void Primitive::clearCachedCounts() const
      {
        m_cachedNumberOfPrimitives = ~0;
        m_cachedNumberOfFaces = ~0;
        m_cachedNumberOfPrimitiveRestarts = ~0;
      }

      inline const VertexAttributeSetSharedPtr & Primitive::getVertexAttributeSet() const
      {
        return( m_vertexAttributeSet );
      }

      inline bool Primitive::isIndexed() const
      {
        return !!m_indexSet;
      }

      inline const IndexSetSharedPtr & Primitive::getIndexSet() const
      {
        return m_indexSet;
      }

      inline PrimitiveType Primitive::getPrimitiveType() const
      {
        return m_primitiveType;
      }

      inline PatchesType Primitive::getPatchesType() const
      {
        return( m_patchesType );
      }

      inline PatchesMode Primitive::getPatchesMode() const
      {
        return( m_patchesMode );
      }

      inline PatchesSpacing Primitive::getPatchesSpacing() const
      {
        return( m_patchesSpacing );
      }

      inline PatchesOrdering Primitive::getPatchesOrdering() const
      {
        return( m_patchesOrdering );
      }

      inline unsigned int Primitive::getInstanceCount() const
      {
        return m_instanceCount;
      }

      inline unsigned int Primitive::getNumberOfVerticesPerPrimitive() const
      {
        switch( getPrimitiveType() )
        {
          case PrimitiveType::LINE_STRIP:
          case PrimitiveType::LINE_LOOP:
          case PrimitiveType::TRIANGLE_STRIP:
          case PrimitiveType::TRIANGLE_FAN:
          case PrimitiveType::QUAD_STRIP:
          case PrimitiveType::POLYGON:
          case PrimitiveType::TRIANGLE_STRIP_ADJACENCY:
          case PrimitiveType::LINE_STRIP_ADJACENCY:
            return 0; // we can't tell

          case PrimitiveType::POINTS:
            return 1;

          case PrimitiveType::LINES:
            return 2;

          case PrimitiveType::TRIANGLES:
            return 3;

          case PrimitiveType::QUADS:
          case PrimitiveType::LINES_ADJACENCY:
            return 4;

          case PrimitiveType::TRIANGLES_ADJACENCY:
            return 6;

          case PrimitiveType::PATCHES:
            return( verticesPerPatch( m_patchesType ) );

          default:
          case PrimitiveType::UNINITIALIZED:
            return ~0; // we can't tell
        }
      }

      inline unsigned int Primitive::getRenderFlags() const
      {
    
        return m_renderFlags;
      }

      inline unsigned int Primitive::getNumberOfPrimitives() const
      {

        // MMM: TODO: add support here for observer to understand if VAS or IS changes
        if( m_cachedNumberOfPrimitives == ~0 )
        {
          // determine face and element count
          determinePrimitiveAndFaceCount();
        }

        return m_cachedNumberOfPrimitives;
      }

      inline unsigned int Primitive::getNumberOfFaces() const
      {

        // MMM: TODO: add support here for observer to understand if VAS or IS changes
        if( m_cachedNumberOfFaces == ~0 )
        {
          // determine face and element count
          determinePrimitiveAndFaceCount();
        }

        return m_cachedNumberOfFaces;
      }

      inline void Primitive::getElementRange( unsigned int & offset, unsigned int & count ) const
      {
        offset = m_elementOffset;
        count  = m_elementCount;  // Note that this can be ~0. Use getElementCount() to query the effective count.
      }

      inline unsigned int Primitive::getElementOffset() const
      {
        return m_elementOffset;
      }

    } // namespace core
  } // namespace sg
} // namespace dp
