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
#include <dp/math/Spherent.h>
#include <dp/util/HashGenerator.h>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/VertexAttribute.h>

namespace dp
{
  namespace sg
  {
    namespace core
    {

      /*! \brief Provides an interface to deal with vertex attribute data.
       *
       * VertexAttributeSet supports generic and conventional vertex attributes. 
       * Conventional vertex attributes are required to support fixed function
       * vertex processing, whereas generic vertex attributes provide a more
       * 'generic' per-vertex computation through vertex programs. Generic
       * attributes do not support fixed function vertex processing! Conventional
       * vertex attributes also support per-vertex computation in a vertex program,
       * but not as general as generic vertex attributes do. For example, the 
       * conventional NORMAL vertex attribute only allows for 3-component values to
       * pass down the vertex pipeline. Other conventional vertex attributes have similar 
       * restrictions. Generic attributes do not have this restriction. */
      class VertexAttributeSet : public Object
      {
        public:
          DP_SG_CORE_API static VertexAttributeSetSharedPtr create();

          DP_SG_CORE_API virtual HandledObjectSharedPtr clone() const;

          DP_SG_CORE_API virtual ~VertexAttributeSet(void);

        public:
      
          enum 
          {
            // conventional vertex attributes
            DP_SG_POSITION = 0,
            DP_SG_VERTEX_WEIGHT,
            DP_SG_NORMAL,
            DP_SG_COLOR,
            DP_SG_SECONDARY_COLOR,
            DP_SG_FOG_COORD,
            DP_SG_UNUSED_1,
            DP_SG_UNUSED_2,
            DP_SG_TEXCOORD0,
            DP_SG_TEXCOORD1,
            DP_SG_TEXCOORD2,
            DP_SG_TEXCOORD3,
            DP_SG_TEXCOORD4,
            DP_SG_TEXCOORD5,
            DP_SG_TEXCOORD6, DP_SG_TANGENT = DP_SG_TEXCOORD6,
            DP_SG_TEXCOORD7, DP_SG_BINORMAL = DP_SG_TEXCOORD7,

            // generic vertex attributes
            DP_SG_ATTR0,
            DP_SG_ATTR1,
            DP_SG_ATTR2,
            DP_SG_ATTR3,
            DP_SG_ATTR4,
            DP_SG_ATTR5,
            DP_SG_ATTR6,
            DP_SG_ATTR7,
            DP_SG_ATTR8,
            DP_SG_ATTR9,
            DP_SG_ATTR10,
            DP_SG_ATTR11,
            DP_SG_ATTR12,
            DP_SG_ATTR13,
            DP_SG_ATTR14,
            DP_SG_ATTR15,
      
            DP_SG_VERTEX_ATTRIB_COUNT = 16

          };

          /*! \brief Exchanges vertex data of two vertex attributes.
           *  \param attrib Indicates the vertex attribute whose data is to be exchanged with
           *  those of the VertexAttribute \a rhs. DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR,
           *  DP_SG_SECONDARY_COLOR, DP_SG_FOG_COORD, DP_SG_TEXCOORD0 - DP_SG_TEXCOORD7, and DP_SG_ATTR0 -
           *  DP_SG_ATTR15 are allowed identifier.
           *  \param rhs The VertexAttribute whose data is to be exchanged with those of the 
           *  vertex attribute indicated by \a attrib. 
           *  \remarks
           *  This function provides an efficient way to exchange data between two vertex attributes. 
           *  It has a constant complexity, that is - it is independent of the amount of vertex data 
           *  hold by the two vertex attributes. 
           *  \n\n
           *  The function does not change the enable state of the vertex attribute indicated by
           *  \a attrib. If \a rhs specifies an empty (e.g. default constructed) VertexAttribute, 
           *  the vertex attribute identified by \a attrib will be empty after this call. 
           *  In that particular case, having the attribute still enabled for rendering, results in
           *  unexpected behavior while rendering. 
           *  \sa setVertexData, setEnabled */
          DP_SG_CORE_API virtual void swapVertexData(unsigned int attrib, VertexAttribute& rhs);

          /*! \brief Reserve space for data in the specified vertex attribute.
           * \param attrib Index to identify the vertex attribute. DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR,
           *  DP_SG_SECONDARY_COLOR, DP_SG_FOG_COORD, DP_SG_TEXCOORD0 - DP_SG_TEXCOORD7, and DP_SG_ATTR0 -
           *  DP_SG_ATTR15 are allowed indices.
           * \param size Specifies the number of coordinates per vertex; must be 1, 2, 3, or 4.
           * \param type Specifies the data type of each coordinate in the input data array. 
           *  In general, symbolic constants dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16,
           *  dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, and dp::DT_FLOAT_64 are accepted.
           * \param count Specifies the number of vertex data to copy.
           * \remarks
           *  This function reserves data in the attribute slot \a attrib. If there already is data in
           *  that slot, \a size and \a type must match the size and type of that data, respectively.
           */
          DP_SG_CORE_API virtual void reserveVertexData( unsigned int attrib, unsigned int size
                                                 , dp::DataType type, unsigned int count );

          /*! \brief Sets data for the specified vertex attribute.
           * \param attrib Index to identify the vertex attribute. DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR,
           *  DP_SG_SECONDARY_COLOR, DP_SG_FOG_COORD, DP_SG_TEXCOORD0 - DP_SG_TEXCOORD7, and DP_SG_ATTR0 -
           *  DP_SG_ATTR15 are allowed indices.
           * \param size Specifies the number of coordinates per vertex; must be 1, 2, 3, or 4.
           * \param type Specifies the data type of each coordinate in the input data array. 
           *  In general, symbolic constants dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16,
           *  dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, and dp::DT_FLOAT_64 are accepted.
           * \param data Specifies the start address of the input data array.
           * \param strideInBytes Specifies the stride between two elements in data. A stride of 0 assumes packed data.
           * \param count Specifies the number of vertex data to copy.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * This function can be used to specify vertex data for generic vertex attributes DP_ATTR0 - DP_SG_ATTR15, 
           * as well as for conventional vertex attributes like DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR, etc..
           * Conventional and generic vertex attributes are aliased, where pairs of conventional and generic
           * attributes are sharing the same storage. 
           * \n\n
           * setVertexData copies data from the specified location and according to the indicated data format.
           * Data that previously was specified for the indicated vertex attribute, or it's respective aliased 
           * attribute, will be released prior to copy the new data.
           * \n\n
           * Conventional vertex attributes are required to support fixed-function vertex processing, whereas 
           * generic vertex attributes provide a more 'generic' per-vertex computation through vertex programs. 
           * Generic attributes do not support fixed-function vertex processing! Conventional vertex attributes 
           * also support per-vertex computation in a vertex program, but not as general as generic vertex 
           * attributes do. Some conventional vertex attributes do have a certain restrictions regarding the
           * number of coordinates per vertex, and the data type of coordinates, as listed in the folowing table:
           * \n\n
           <TABLE ALIGN=center CELLSPACING=0 CELLPADDING=0 BORDER=0>
           <TR BGCOLOR="#000000">
           <TD>
           <TABLE CELLSPACING=1 CELLPADDING=2 BORDER=0>
           <TR BGCOLOR="#4040c0">
           <TD ><b><font size=+1 color="#ffffff"> Attribute </font></b></TD>
           <TD ><b><font size=+1 color="#ffffff"> Size </font></b></TD>
           <TD ><b><font size=+1 color="#ffffff"> Type </font></b></TD>
           </TR>
           <TR BGCOLOR="#ffffff">
           <TD>DP_SG_POSITION</TD>  
           <TD>3</TD>
           <TD>dp::DT_FLOAT_32</TD>
           </TR>
           <TR BGCOLOR="#ffffff">
           <TD>DP_SG_NORMAL</TD>  
           <TD>3</TD>
           <TD>dp::DT_FLOAT_32</TD>
           </TR>
           <TR BGCOLOR="#ffffff">
           <TD>DP_SG_COLOR</TD>  
           <TD>3, 4</TD>
           <TD>dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16, 
               dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, dp::DT_FLOAT_64 </TD>
           </TR>
           <TR BGCOLOR="#ffffff">
           <TD>DP_SG_SECONDARY_COLOR</TD>  
           <TD>3</TD>
           <TD>dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16, 
           dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, dp::DT_FLOAT_64 </TD>
           </TR>
           <TR BGCOLOR="#ffffff">
           <TD>DP_SG_FOG_COORD</TD>  
           <TD>1</TD>
           <TD>dp::DT_FLOAT_32, dp::DT_FLOAT_64</TD>
           </TR>
           <TR BGCOLOR="#ffffff">
           <TD>DP_SG_TEXCOORD[0-7]</TD>  
           <TD>1, 2, 3, 4</TD>
           <TD>dp::DT_INT_16, dp::DT_INT_32, dp::DT_FLOAT_32, dp::DT_FLOAT_64</TD>
           </TR>
           </TABLE>
           </TD>
           </TR>
           </TABLE>
           * \n\n
           * The data copied is not automatically enabled for rendering! Enabling the copied data for rendering
           * requires an additional call to setEnabled.
           * \sa setEnabled
           * */
          DP_SG_CORE_API virtual void setVertexData( unsigned int attrib, unsigned int size, dp::DataType type
                                             , const void * data, unsigned int strideInBytes, unsigned int count
                                             , bool enable = true );
      
          /*! \brief Overwrites data for the specified vertex attribute starting at the indicated position.
           * \param attrib Index to identify the vertex attribute. DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR,
           *  DP_SG_SECONDARY_COLOR, DP_SG_FOG_COORD, DP_SG_TEXCOORD0 - DP_SG_TEXCOORD7, and DP_SG_ATTR0 -
           *  DP_SG_ATTR15 are allowed indices.
           * \param pos Marks the position of the first vertex inside the already specified array of vertex data,
           *  where overwriting should start. If the magic value ~0 is specified, the input data will be appended
           *  to the array of vertex data.
           * \param size Specifies the number of coordinates per vertex; must be 1, 2, 3, or 4.
           * \param type Specifies the data type of each coordinate in the input data array. 
           *  In general, symbolic constants dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16,
           *  dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, and dp::DT_FLOAT_64 are accepted.
           * \param data Specifies the start address of the input data array.
           * \param strideInBytes Specifies the stride between two elements in data. A stride of 0 assumes packed data.
           * \param count Specifies the number of vertex data to copy.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * This setVertexData overload re-specifies previously copied vertex data starting at the indicated 
           * position. If the magic value ~0 is specified for \a pos, the input vertex data will be appended
           * to the previously copied vertex data. */
          DP_SG_CORE_API virtual void setVertexData( unsigned int attrib, unsigned int pos, unsigned int size
                                             , dp::DataType type, const void * data, unsigned int strideInBytes
                                             , unsigned int count, bool enable = true );

          /*! \brief Overwrites data for the specified vertex attribute using the provided buffer information.
           * \param attrib Index to identify the vertex attribute. DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR,
           *  DP_SG_SECONDARY_COLOR, DP_SG_FOG_COORD, DP_SG_TEXCOORD0 - DP_SG_TEXCOORD7, and DP_SG_ATTR0 -
           *  DP_SG_ATTR15 are allowed indices.
           * \param size Specifies the number of coordinates per vertex; must be 1, 2, 3, or 4.
           * \param type Specifies the data type of each coordinate in the input data array. 
           *  In general, symbolic constants dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16,
           *  dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, and dp::DT_FLOAT_64 are accepted.
           * \param buffer The buffer containing the vertex attribute data.
           * \param offset The offset into the buffer, marking the start of the vertex attribute data.
           * \param strideInBytes The stride between two consecutive vertex attribute data, in bytes.
           * \param count Specifies the number of vertex data.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * This setVertexData overload re-specifies the previously copied vertex data by abandoning the old
           * buffer and referencing the new buffer. */
          DP_SG_CORE_API virtual void setVertexData( unsigned int attrib, unsigned int size, dp::DataType type
                                             , const BufferSharedPtr &buffer, unsigned int offset
                                             , unsigned int strideInBytes, unsigned int count
                                             , bool enable = true );
      
          /*! \brief Overwrites data referenced to by an index array.
           * \param attrib Index to identify the vertex attribute. DP_SG_POSITION, DP_SG_NORMAL, DP_SG_COLOR,
           *  DP_SG_SECONDARY_COLOR, DP_SG_FOG_COORD, DP_SG_TEXCOORD0 - DP_SG_TEXCOORD7, and DP_SG_ATTR0 -
           *  DP_SG_ATTR15 are allowed indices.
           * \param to Specifies the start address of the index array, that will be taken
           *  to reference the respective vertex data to be re-specified. If NULL is passed here,
           *  the input vertex data will be appended to the array of vertex data that previously was specified.
           * \param from Specifies the start address of the index array, that will be taken to reference
           * the input vertex data to overwrite the previously specified vertex data.
           * \param size Specifies the number of coordinates per vertex; must be 1, 2, 3, or 4.
           * \param type Specifies the data type of each coordinate in the input data array. 
           *  In general, symbolic constants dp::DT_INT_8, dp::DT_UNSIGNED_INT_8, dp::DT_INT_16, dp::DT_UNSIGNED_INT_16,
           *  dp::DT_INT_32, dp::DT_UNSIGNED_INT_32, dp::DT_FLOAT_32, and dp::DT_FLOAT_64 are accepted.
           * \param data Specifies the start address of the input data array.
           * \param stride Stride between two elements in data. Use 0 for packed data.
           * \param count Specifies the number of vertex data to copy.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * This setVertexData overload copies \a count vertex data from the vertex data array pointed
           * to by \a data. Vertex data that previously was specified for the indicated vertex attribute, 
           * or it's respective aliased attribute, will be released prior to copy the new data.
           * The input vertex data will not be copied as an contiguous range, but by 
           * referencing the input vertex data through the index array specified by \a from. 
           * The destination vertex data, in the same way, will be accessed through the index array 
           * pointed to by \a to. The following pseudo code illustrates how the copying proceeds:
           * \code
           *      for ( unsigned int i = 0; i < count; ++i )    
           *      {
           *        destination[to[i]] = source[from[i]];
           *      }
           * \endcode
           * \n\n
           * If a NULL pointer is passed through \a to, the input vertex data will be appended to the 
           * vertex data that was copied prior to this call.
           */
          DP_SG_CORE_API virtual void setVertexData( unsigned int attrib, const unsigned int * to
                                             , const unsigned int * from, unsigned int size
                                             , dp::DataType type, const void * data, unsigned int stride
                                             , unsigned int count, bool enable = true );
      
          /*! \brief Removes vertex data assigned to the indicated vertex attribute.
           * \param attrib Specifies the vertex attribute for which to remove previously copied vertex data.
           * \param disable Optional bool to disable this vertex attribute. Default is \c true.
           * \remarks 
           * The function releases the vertex data that previously was specified for the vertex attribute
           * indicated by \a attrib. 
           * \note
           * The function does not reset the enable state for the indicated vertex attribute!
           * If client code intends to permanently remove vertex data for the indicated vertex attribute,
           * the vertex attribute must no longer be enabled! This can be achieved through an explicit call to
           * VertexAttributeSet::setEnabled with \c false as second argument. Leaving the vertex attribute 
           * enabled results in unexpected behavior while rendering.
           * \sa setEnable */
          DP_SG_CORE_API virtual void removeVertexData( unsigned int attrib, bool disable = true );

          /*! \brief Retrieves the vertex data for the indicated vertex attribute.
           * \param attrib Specifies the vertex attribute from which to get the vertex data.
           * \return The function returns a type-less pointer to the vertex data that was assigned to 
           * the vertex attribute indicated by \a attrib prior to this call. If for this vertex attribute
           * no vertex data was specified before, the function returns NULL. 
           * \remarks
           * The function returns a type-less pointer to the stored vertex array. To access the vertex data
           * from this pointer it requires the determination of additional vertex data traits: Use 
           * getTypeOfVertexData to determine the per-coordinate type for each vertex in the vertex array,
           * getSizeOfVertexData retrieves the number of coordinates per vertex, and getNumberOfVertexData
           * returns the number of vertices in the vertex array. */
          DP_SG_CORE_API virtual Buffer::DataReadLock getVertexData(unsigned int attrib) const;

          /*! \brief Retrieve the BufferSharedPtr corresponding to the given attribute
           * \param attrib Specifies the vertex attribute from which to get the vertex data.
           * \return The Buffer for the given attribute. It is invalid if no data was specified before.
          **/
          DP_SG_CORE_API virtual const BufferSharedPtr & getVertexBuffer(unsigned int attrib) const;

          template <typename ValueType>
          typename Buffer::ConstIterator<ValueType>::Type getVertexData(unsigned int attrib) const;

          /*! \brief Returns the number of coordinates per vertex.
           * \param attrib Indicates the vertex attribute to query. 
           * \return The function returns the number of coordinates per vertex stored for the indicated attribute.
           * This number can be 1, 2, 3, or 4. If no vertex data was specified for this vertex attribute before,
           * the function returns 0. */
          DP_SG_CORE_API virtual unsigned int getSizeOfVertexData(unsigned int attrib) const;

          /*! \brief Returns the type identifier of each coordinate.
           * \param attrib Indicates the vertex attribute to query. 
           * \return The function returns a symbolic constant indicating the type of each coordinate in the 
           * vertex array stored for the specified vertex attribute. If no vertex data was specified for this
           * vertex attribute before, the function returns dp::DT_UNKNOWN.
           * \remarks
           * The function returns the per-coordinate type specifier that was used with a corresponding call
           * to setVertexData. Use getSizeOfVertexData to query the number of coordinates for each vertex stored.
           * You can use the dp::getSizeOf convenient function to determine the size in bytes of the type 
           * returned by this function.
           * \sa setVertexData */
          DP_SG_CORE_API virtual dp::DataType getTypeOfVertexData(unsigned int attrib) const;
      
          /*! \brief Returns the size of the vertex array stored for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute to query. 
           * \return The function returns the number of vertices stored for the indicated vertex attribute. */
          DP_SG_CORE_API virtual unsigned int getNumberOfVertexData(unsigned int attrib) const;

          /*! \brief Returns the stride between two elements of the vertex array for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute to query.
           * \return The stride in bytes between two elements in the indicated vertex attribute.
           */
          DP_SG_CORE_API virtual unsigned int getStrideOfVertexData( unsigned int attrib ) const;

          /*! \brief Returns the offset of the first element of the vertex array for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute to query.
           * \return The offset in bytes of the first element
           */
          DP_SG_CORE_API virtual unsigned int getOffsetOfVertexData( unsigned int attrib ) const;

         /*! \brief Returns whether the vertex attribute is contiguous.
           * \param attrib Indicates the vertex attribute to query.
           * \return Returns \c true when stride matches vertex byte size
           */
          DP_SG_CORE_API virtual bool isContiguousVertexData( unsigned int attrib ) const;

          /*! \brief Specifies the enable state for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute for which to specify the enable state. 
           * \param enable Appoints the enable state for the indicated vertex attribute.
           * \remarks
           * Vertex data that were specified for the indicated vertex attribute will not automatically
           * be enabled for rendering. Use this function to appoint the enable state for the indicated
           * vertex attribute. If \c true is passed as second argument, the corresponding vertex array
           * will be enabled for rendering. If \c false is passed instead, the vertex array will be
           * explicitly disabled for rendering. 
           * \sa isEnabled */
          DP_SG_CORE_API void setEnabled(unsigned int attrib, bool enable);

          /*! \brief Retrieves the enable state for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute to query. 
           * \return The function returns \c true, if the indicated vertex attribute is enabled for rendering.
           * Otherwise the function returns \c false. */  
          DP_SG_CORE_API bool isEnabled(unsigned int attrib) const;

          /*! \brief Specifies the normalize-enable state for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute for which to specify the normalize-enable state. 
           * \param enable Appoints the normalize-enable state for the indicated vertex attribute.
           * \remarks
           * This function is meaningful only for generic vertex attributes, that is - DP_ATTR0 - DP_SG_ATTR15.
           * This function can be used to indicate that fixed-point generic vertex attribute array components
           * should be converted to [-1,1] or [0,1] ranges, for signed and unsigned types resp.. 
           * If you pass \c true as second argument, normalization will be implicitly enabled for the
           * vertex data of the indicated vertex attribute. The normalize request will be forwarded to the
           * renderer. The conversion will not be performed on client-site data! 
           * \n\n
           * Normalize initially is disabled. */
          DP_SG_CORE_API void setNormalizeEnabled(unsigned int attrib, bool enable); 

          /*! \brief Retrieves the normalize-enable state for the indicated vertex attribute.
           * \param attrib Indicates the vertex attribute to query. 
           * \return The function returns \c true, if normalize conversion is enabled for the indicated 
           * vertex attribute. Otherwise, the function returns \c false. */
          DP_SG_CORE_API bool isNormalizeEnabled(unsigned int attrib) const; 

          /*! \brief Sets position data for this VertexAttributeSet.
           * \param vertices Specifies the array of input positions. 
           * \param count indicates the number of vertices to copy. 
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to specify position data for this VertexAttributeSet. 
           * A single position is required to have three coordinates of type float. 
           * Positions that previously were specified for this VertexAttributeSet will 
           * be released prior to copying the new positions.
           * \n\n
           * The vertices initially are enable for rendering if specified using this function.
           * Alternatively, you can use setVertexData with DP_SG_POSITION as first argument
           * to specify positions, but this would require an explicit enable through setEnabled.
           * \sa setVertexData, setEnabled */ 
          DP_SG_CORE_API void setVertices( const dp::math::Vec3f * vertices, unsigned int count, bool enable = true );

          /*! \brief Overwrites position data for this VertexAttributeSet starting at the indicated position.
          * \param pos Marks the zero-based index of the first vertex inside the already specified array of positions,
          *  where overwriting should start. If the magic value ~0 is specified, the input data will be appended
          *  to the array of position data.
          * \param vertices Specifies the array of input positions. 
          * \param count indicates the number of vertices to copy. 
          * \param enable Optional bool to enable this vertex attribute. Default is \c true.
          * \remarks
          * Use this setVertices overload to overwrite or to append position data. 
          * A single position is required to have three coordinates of type float. 
          * \sa setVertexData */ 
          DP_SG_CORE_API void setVertices( unsigned int pos, const dp::math::Vec3f * vertices, unsigned int count, bool enable = true );

          /*! \brief Retrieves position data from the VertexAttributeSet.
           * \return The function returns the address of the first position stored in the array.
           * \note If there are no vertices, or the data is specified by the generic vertex attribute
           *  DP_ATTR0 and doesn't fit to the usual vertex size and format of three floats per vertex,
           *  NULL is returned. */
          DP_SG_CORE_API Buffer::ConstIterator<dp::math::Vec3f>::Type getVertices() const;

          /*! \brief Returns the size of the position array.
           * \return The function returns the number of positions stored for this VertexAttributeSet. */
          DP_SG_CORE_API unsigned int getNumberOfVertices() const;

          /*! \brief Returns the byte-size requirement per vertex.
           * \return Returns the number of bytes required to store a single vertex and associated attributes (normals, etc.). */
          DP_SG_CORE_API unsigned int getPerVertexBytes() const;

          // no -longer supported
          //DP_SG_CORE_API void setVertexWeights(const float * weights, unsigned int count);
          //DP_SG_CORE_API void setVertexWeights(unsigned int pos, const float * weights, unsigned int count);
          //DP_SG_CORE_API const void * getVertexWeights() const;
          //DP_SG_CORE_API unsigned int getNumberOfVertexWeights() const;

          /*! \brief Sets normals for this VertexAttributeSet.
           * \param normals Specifies the normal array.
           * \param count Indicates the size of the input normal array. 
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to specify normals for this VertexAttributeArray.
           * Normals that previously were specified for this VertexAttributeSet will be released prior
           * to copying the new normals.
           * \n\n
           * The normals initially are enable for rendering if specified using this function.
           * Alternatively, you can use setVertexData with DP_SG_NORMAL as first argument
           * to specify normals, but this would require an explicit enable through setEnabled. 
           * \sa setVertexData, setEnabled */
          DP_SG_CORE_API void setNormals( const dp::math::Vec3f * normals, unsigned int count, bool enable = true );

          /*! \brief Overwrites normals for this VertexAttributeSet starting at the indicated position.
           * \param pos Marks the zero-based index of the first vertex inside the already specified array of normals,
           *  where overwriting should start. If the magic value ~0 is specified, the input data will be appended
           *  to the array of normals.
           * \param normals Specifies the normal array.
           * \param count Indicates the size of the input normal array. 
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this setNormals overload to overwrite or to append normal data. 
           * \sa setVertexData */ 
          DP_SG_CORE_API void setNormals( unsigned int pos, const dp::math::Vec3f * normals, unsigned int count, bool enable = true );
      
          /*! \brief Retrieves normals from the VertexAttributeSet.
           * \return The function returns the address of the first normal stored in the array.
           * \note If there are no normals, or the data is specified by the generic vertex attribute
           *  DP_SG_ATTR2 and doesn't fit to the usual normal size and format of three floats per vertex,
           *  NULL is returned. */
          DP_SG_CORE_API Buffer::ConstIterator<dp::math::Vec3f>::Type getNormals() const;

          /*! \brief Returns the size of the normal array.
           * \return The function returns the number of normals stored for this VertexAttributeSet. */
          DP_SG_CORE_API unsigned int getNumberOfNormals() const;

          /*! \brief Sets vertex colors for this VertexAttributeSet.
           * \param colors Specifies the input color array.
           * \param count Indicates the size of the input color array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to specify colors for this VertexAttributeSet. 
           * Colors can be specified to have N=3, and N=4 components of type T=char, T=unsigned char,
           * T=short, T=unsigned short, T=int, T=unsigned int, T=float, and T=double. 
           * Colors that previously were specified for this VertexAttributeSet will be released prior
           * to copying the new colors. 
           * \n\n
           * The colors initially are enable for rendering if specified using this function.
           * Alternatively, you can use setVertexData with DP_SG_COLOR as first argument
           * to specify vertex colors, but this would require an explicit enable through setEnabled. 
           * \sa setVertexData, setEnabled */
          template <unsigned int N, typename T> void setColors( const dp::math::Vecnt<N,T> * colors, unsigned int count, bool enable = true );

          /*! \brief Overwrites colors for this VertexAttributeSet starting at the indicated position.
           * \param pos Marks the zero-based index of the first color inside the already specified color array,
           *  where overwriting should start. If the magic value ~0 is specified, the input data will be appended
           *  to the array of colors.
           * \param colors Specifies the input color array.
           * \param count Indicates the size of the input color array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to overwrite or append colors for this VertexAttributeSet. 
           * Colors can be specified to have N=3, and N=4 components of type T=char, T=unsigned char,
           * T=short, T=unsigned short, T=int, T=unsigned int, T=float, and T=double. 
           * Specifying a different number of color components (N), or a different type (T) than specified
           * with a previous setColors call results in undefined behavior! */
          template <unsigned int N, typename T> void setColors( unsigned int pos, const dp::math::Vecnt<N,T> * colors, unsigned int count, bool enable = true );

          /*! \brief Retrieves colors from the VertexAttributeSet.
           * \return The function returns the address of the first color stored in the array. */
          template <typename ValueType> typename Buffer::ConstIterator<ValueType>::Type getColors() const;

          /*! \brief Returns the size of the color array.
           * \return The function returns the number of colors stored for this VertexAttributeSet. */
          DP_SG_CORE_API unsigned int getNumberOfColors() const;

          /*! \brief Returns the number of components per color.
           * \return The function returns the number of components per color as specified with a 
           * previous call to setColors. */
          DP_SG_CORE_API unsigned int getSizeOfColors() const;
   
          /*! \brief Returns the type of each color component.
           * \return The function returns a symbolic constant representing the type of each component 
           * of the currently stored colors. */
          DP_SG_CORE_API unsigned int getTypeOfColors() const;

          /*! \brief Sets secondary colors for this VertexAttributeSet.
           * \param colors Specifies the input color array.
           * \param count Indicates the size of the input color array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to specify secondary colors for this VertexAttributeSet. 
           * Secondary colors can be specified as three-component colors, of type T=char, T=unsigned char,
           * T=short, T=unsigned short, T=int, T=unsigned int, T=float, and T=double. 
           * Secondary colors that previously were specified for this VertexAttributeSet will be released prior
           * to copying the new colors.
           * \n\n
           * The secondary colors initially are enable for rendering if specified using this function.
           * Alternatively, you can use setVertexData with DP_SECONDARY_COLOR as first argument
           * to specify secondary colors, but this would require an explicit enable through setEnabled. 
           * \sa setVertexData, setEnabled */
          template <typename T> void setSecondaryColors( const dp::math::Vecnt<3,T> * colors, unsigned int count, bool enable = true );

          /*! \brief Overwrites colors for this VertexAttributeSet starting at the indicated position.
           * \param pos Marks the zero-based index of the first color inside the already specified color array,
           *  where overwriting should start. If the magic value ~0 is specified, the input data will be appended
           *  to the array of colors.
           * \param colors Specifies the input color array.
           * \param count Indicates the size of the input color array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to overwrite or append secondary colors for this VertexAttributeSet. 
           * Secondary colors can be specified as three-component colors, of type ValueType=char, ValueType=unsigned char,
           * ValueType=short, ValueType=unsigned short, ValueType=int, ValueType=unsigned int, ValueType=float, and ValueType=double. 
           * Specifying a different type (ValueType) than specified with a previous setSecondaryColors call results 
           * in undefined behavior! */
          template <typename ValueType> void setSecondaryColors( unsigned int pos, const dp::math::Vecnt<3,ValueType> * colors, unsigned int count, bool enable = true );
      
          /*! \brief Retrieves secondary colors from the VertexAttributeSet.
           * \return The function returns the address of the first secondary color stored in the array. */
          template <typename ValueType> typename Buffer::ConstIterator<ValueType>::Type getSecondaryColors() const;

          /*! \brief Returns the size of the secondary color array.
           * \return The function returns the number of secondary colors stored for this VertexAttributeSet. */
          DP_SG_CORE_API unsigned int getNumberOfSecondaryColors() const;
      
          /*! \brief Returns the type of each secondary color component.
          * \return The function returns a symbolic constant representing the type of each component 
          * of the currently stored secondary colors. */
          DP_SG_CORE_API unsigned int getTypeOfSecondaryColors() const;

          /*! \brief Sets fog coordinates for this VertexAttributeSet.
           * \param coords Specifies the input coordinate array.
           * \param count Indicates the size of the input coordinate array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to specify fog coordinates for this VertexAttributeSet. 
           * Fog coordinates can be specified as single-component coordinates, of type T=float, and T=double. 
           * Fog coordinates that previously were specified for this VertexAttributeSet will be released prior
           * to copying the new coordinates.
           * \n\n
           * The fog coordinates initially are enable for rendering if specified using this function.
           * Alternatively, you can use setVertexData with DP_FOG_COORD as first argument
           * to specify fog coordinates, but this would require an explicit enable through setEnabled. 
           * \sa setVertexData, setEnabled */
          template <typename T> void setFogCoords( const T * coords, unsigned int count, bool enable = true );
      
          /*! \brief Overwrites fog coordinates starting at the indicated position.
           * \param pos Marks the zero-based index of the first coordinate inside the already specified fog coordinate
           * array, where overwriting should start. If the magic value ~0 is specified, the input data will be appended
           * to the array of fog coordinates.
           * \param coords Specifies the input coordinate array.
           * \param count Indicates the size of the input coordinate array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to overwrite or append fog coordinates. 
           * Fog coordinates can be specified as single-component coordinates, of type T=float, and T=double. 
           * Specifying a different type (T) than specified with a previous setFogCoords call results 
           * in undefined behavior! */
          template <typename T> void setFogCoords( unsigned int pos, T * coords, unsigned int count, bool enable = true );

          /*! \brief Retrieves fog coordinates from the VertexAttributeSet.
           * \return The function returns the address of the first coordinate stored in the array.
           * \note If there are no fog coords, or the data is specified by the generic vertex attribute
           *  DP_SG_ATTR5 and doesn't fit to the usual fog coord size of one component per vertex, NULL
           *  is returned. */
          Buffer::ConstIterator<float>::Type getFogCoords() const;

          /*! \brief Returns the size of the fog coordinate array.
           * \return The function returns the number of fog coordinates stored for this VertexAttributeSet. */
          DP_SG_CORE_API unsigned int getNumberOfFogCoords() const;

          /*! \brief Returns the type of the fog coordinates.
           * \return The function returns a symbolic constant representing the type of the currently stored 
           * fog coordinates. */
          DP_SG_CORE_API unsigned int getTypeOfFogCoords() const;

          /*! \brief Sets a texture coordinate array.
           * \param unit Specifies the texture unit for which to specify the texture coordinate array.
           * \param coords Specifies the input coordinate array.
           * \param count Indicates the size of the input coordinate array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to specify a texture coordinate array for the indicated texture unit. 
           * An array element can be specified to have N=1, N=2, N=3, and N=4 coordinates of type 
           * T=short, T=int, T=float, or T=double. 
           * Texture coordinates that previously were specified for the indicated
           * texture unit will be released prior to copying the new texture coordinates.
           * \n\n
           * The texture coordinates initially are enable for rendering if specified using this function.
           * Alternatively, you can use setVertexData with DP_SG_TEXCOORD0+unit as first argument
           * to specify texture coordinates, but this would require an explicit enable through setEnabled. 
           * \sa setVertexData, setEnabled */
          template <unsigned int N, typename T> void setTexCoords( unsigned int unit, const dp::math::Vecnt<N,T> * coords, unsigned int count, bool enable = true );

          /*! \brief Overwrites texture coordinates starting at the indicated position.
           * \param unit Specifies the texture unit for which to specify the texture coordinates.
           * \param pos Marks the zero-based index of the first coordinate inside the already specified texture coordinate
           * array, where overwriting should start. If the magic value ~0 is specified, the input data will be appended
           * to the array of texture coordinates.
           * \param coords Specifies the input coordinate array.
           * \param count Indicates the size of the input coordinate array.
           * \param enable Optional bool to enable this vertex attribute. Default is \c true.
           * \remarks
           * Use this function to overwrite or append texture coordinates for the indicated texture unit. 
           * Texture coordinates can be specified to have N=1, N=2, N=3, and N=4 components of type 
           * T=short, T=int, T=float, and T=double. 
           * Specifying a different number of components (N), or a different type (T) than specified
           * with a previous setTexCoords call results in undefined behavior! */
          template <unsigned int N, typename T> void setTexCoords( unsigned int unit, unsigned int pos, const dp::math::Vecnt<N,T> * coords, unsigned int count, bool enable = true );

          /*! \brief Retrieves texture coordinates set for the indicated texture unit.
           * \param unit Specifies the texture unit to query.
           * \return The function returns the address of the first coordinate stored in the array. */
          template <typename ValueType> typename Buffer::ConstIterator<ValueType>::Type getTexCoords(unsigned int unit) const;

          /*! \brief Returns the size of the texture coordinate array.
           * \param unit Specifies the texture unit to query.
           * \return The function returns the number of texture coordinate array elements stored for the 
           * indicated texture unit. */
          DP_SG_CORE_API unsigned int getNumberOfTexCoords(unsigned int unit) const;

          /*! \brief Returns the number of coordinates per texture coordinate array element.
           * \param unit Specifies the texture unit to query. 
           * \return The function returns the number of coordinates per texture coordinate array element. */
          DP_SG_CORE_API unsigned int getSizeOfTexCoords(unsigned int unit) const;

          /*! \brief Returns the type of each texture coordinate.
           * \param unit Specifies the texture unit to query. 
           * \return The function returns a symbolic constant representing the type of each coordinate
           * in an element of the texture coordinate array currently assigned to the indicated texture unit. */
          DP_SG_CORE_API unsigned int getTypeOfTexCoords(unsigned int unit) const;

          /*! \brief Update a given VertexAttribute
           *  \param attrib Indicates the vertex attribute whose data is to be exchanged with
           *  \param vertexAttribute VertexAttribute which should be copied over the given attrib
           *  \param enable Optional bool to enable this vertex attribute. Default is \c true.
           *  \remarks The given attribute will not be enabled. Call \sa setEnabled to enable the attribute
           */
          DP_SG_CORE_API void setVertexAttribute( unsigned int attrib, const VertexAttribute &vertexAttribute, bool enable = true );

          /*! \brief Fetch data from one VertexAttribute.
           *  \param attrib Indicates the vertex attribute whose data is to be exchanged with
           *  \return A const reference to the given vertex attribute.
           */
          DP_SG_CORE_API const VertexAttribute &getVertexAttribute( int attrib ) const;

          /*! \brief Combines the data buffers of all attributes into a single data buffer.
          **/
          DP_SG_CORE_API void combineBuffers();

          //! Assignment operator
          /** Performs a deep copy. */
          DP_SG_CORE_API VertexAttributeSet & operator=(const VertexAttributeSet & rhs);

          //! Test for equivalence with an other VertexAttributeSet.
          /** \return true if the VertexAttributeSet \a p is equivalent to this VertexAttributeSet. */
          DP_SG_CORE_API virtual bool isEquivalent( ObjectSharedPtr const& object, bool ignoreNames, bool deepCompare ) const;

          REFLECTION_INFO_API( DP_SG_CORE_API, VertexAttributeSet );
        protected:
          //! Default-constructs an empty VertexAttributeSet.
          DP_SG_CORE_API VertexAttributeSet();

          //! Copy Constructor.
          DP_SG_CORE_API VertexAttributeSet( const VertexAttributeSet &rhs );

          /*! \brief Feed the data of this object into the provided HashGenerator.
           *  \param hg The HashGenerator to update with the data of this object.
           *  \sa getHashKey */
          DP_SG_CORE_API virtual void feedHashGenerator( dp::util::HashGenerator & hg ) const;

        private:
          typedef std::map<int,VertexAttribute> AttributeContainer;

        private:
          void subscribeBuffer( const AttributeContainer::const_iterator & it );
          void unsubscribeBuffer( const AttributeContainer::const_iterator & it );

        private:
          static DP_SG_CORE_API VertexAttribute m_emptyAttribute;

          /** \brief Called by BufferObserver if a the buffer for the vertices has been changed **/
          void onBufferChanged();

          /** \brief Observe a buffer and notify the VertexAttributeSet upon changes.
                     This class is designed to observe only the buffer for vertices since
                     those can change the bounding volume.
          **/
          class BufferObserver : public dp::util::Observer
          {
          public:
            void setVertexAttributeSet( VertexAttributeSet *vertexAttributeSet )
            {
              m_vertexAttributeSet = vertexAttributeSet;
            }

            virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload )
            {
              m_vertexAttributeSet->onBufferChanged();
            }

            virtual void onDestroyed( const Subject& subject, dp::util::Payload* payload )
            {
              m_vertexAttributeSet = nullptr;
            }

          private:
            VertexAttributeSet* m_vertexAttributeSet;
          };

        private:
          unsigned int        m_enableFlags; // bits 0-15 conventional attributes, bits 16-31 generic aliases
          unsigned int        m_normalizeEnableFlags; // only for generic attributes
          AttributeContainer  m_vattribs;

          BufferObserver      m_bufferObserver;
      };


      DP_SG_CORE_API void copy( const VertexAttributeSetSharedPtr & src, const VertexAttributeSetSharedPtr & dst );

      /*! \brief Copy selected vertices from one VertexAttributeSet to another.
       *  \param from The VertexAttributeSet to copy from.
       *  \param to The VertexAttributeSet to copy to.
       *  \param indices A reference to a vector of indices pointing into \a from, that determine the vertices to copy.
       *  After the copy operation, it holds the corresponding indices pointing into \a to.
       *  \remarks This function is used to copy selected vertices of IndependentPrimitiveSets */
      DP_SG_CORE_API void copySelectedVertices( const VertexAttributeSetSharedPtr & from
                                        , const VertexAttributeSetSharedPtr & to
                                        , std::vector<unsigned int> & indices );

      /*! \brief Copy selected vertices from one VertexAttributeSet to another.
       *  \param from The VertexAttributeSet to copy from.
       *  \param to The VertexAttributeSet to copy to.
       *  \param indices A reference to a vector of indices into \a from, that determine the vertices to copy. After the
       *  copy operation, it holds the corresponding indices into \a to.
       *  \remarks This function is used to copy selected vertices of StrippedPrimitiveSets */
      DP_SG_CORE_API void copySelectedVertices( const VertexAttributeSetSharedPtr & from
                                        , const VertexAttributeSetSharedPtr & to
                                        , std::vector<std::vector<unsigned int> > &indices );

      //! Type to specify the type of texture coordinate to generate.
      /** There are three different types of texture coordinate generation available: cylindrical, planar, and spherical.*/
      typedef enum
      {
        TCT_CYLINDRICAL,      //!<  Puts a cylinder around the \c BoundingSphere with the z-axis as it's axis and projects the vertices of the \c VertexAttributeSet on it to get the texture coordinates.
        TCT_PLANAR,           //!<  Puts a plane in the x-y-plane and vertically projects the vertices of the \c VertexAttributeSet on that plane to the the texture coordinates.
        TCT_SPHERICAL,        //!<  Projects the vertices of the \c VertexAttributeSet radially on the \c BoundingSphere to get the texture coordinates.
      } TextureCoordType;

      /*! \relates VertexAttributeSet
       *  Generate texture coordinates into a texture unit.
       *  Calculates either spherical, cylindrical, or planar 
       *  two-dimensional texture coordinates into texture unit \a
       *  tu. The \a sphere can be either the bounding sphere of 
       *  this Primitive to calculate local texture coordinates or
       *  some other for non-local texture coordinates.        */
      DP_SG_CORE_API void generateTexCoords( VertexAttributeSetSharedPtr const& vas       //!< pointer to the VertexAttributeSet to use
                                     , TextureCoordType tct           //!< type of texture coordinates to generate
                                     , const dp::math::Sphere3f &sphere //!< bounding sphere to map coordinates to
                                     , unsigned int tc = VertexAttributeSet::DP_SG_TEXCOORD0   //!< vertex attrib to generate the coordinates in
                                     );

      /*! \brief Determine if an attribute identifier identifies a generic or a conventional attribute.
       *  \param attrib The attribute identifier to test.
       *  \return \c true if \a attrib identifies a generic attribute, otherwise \c false.
       *  \sa attribIndex */
      bool genericAttrib( unsigned int attrib );

      /*! \brief Translates an attribute identifier to the corresponding conventional attribute identifier.
       *  \param attrib The attribute identifier to translate.
       *  \return The conventional attribute identifier corresponding to \a attrib.
       *  \remarks If \a attrib already identifies a conventional attribute, it is returned unchanged.
       *  Otherwise it is a generic attribute identifier and the corresponding conventional one is
       *  returned.
       *  \sa genericAttrib */
      unsigned int attribIndex( unsigned int attrib );

      template <typename ValueType>
      typename Buffer::ConstIterator<ValueType>::Type VertexAttributeSet::getVertexData(unsigned int attrib) const
      {
        DP_ASSERT( DP_SG_POSITION<=attrib && attrib<=DP_SG_ATTR15 ); // undefined behavior on invalid attrib

        // we alias generic and conventional vertex attributes, that is - 
        // pairs of generic and conventional vertex attributes are sharing the same storage
        unsigned int attrIndex = attribIndex(attrib);
        AttributeContainer::const_iterator it = m_vattribs.find( attrIndex );
        if ( it == m_vattribs.end() )
        {
          return Buffer::ConstIterator<ValueType>::Type();
        }
        else
        {
          Buffer::DataReadLock lock = getVertexData( it->first );
          return( typename Buffer::ConstIterator<ValueType>::Type( lock.getPtr<ValueType>(), it->second.getVertexDataStrideInBytes(), lock ) );
        }
      }

      inline void VertexAttributeSet::setEnabled(unsigned int attrib, bool enable)
      {
        DP_ASSERT(attrib>=DP_SG_POSITION && attrib<=DP_SG_ATTR15);
        m_enableFlags &= ~(1<<attrib);
        m_enableFlags |= ((!!enable)<<attrib);
        notify( Event( this ) );
      }

      inline bool VertexAttributeSet::isEnabled(unsigned int attrib) const
      {
        DP_ASSERT(attrib>=DP_SG_POSITION && attrib<=DP_SG_ATTR15);
        return !!(m_enableFlags & (1<<attrib));
      }

      inline void VertexAttributeSet::setNormalizeEnabled(unsigned int attrib, bool enable)
      {
        DP_ASSERT(attrib>=DP_SG_ATTR0 && attrib<=DP_SG_ATTR15); // only for generic attributes!
        m_normalizeEnableFlags &= ~(1<<attrib);
        m_normalizeEnableFlags |= ~((!!enable)<<attrib);
        notify( Event( this ) );
      }

      inline bool VertexAttributeSet::isNormalizeEnabled(unsigned int attrib) const
      {
        DP_ASSERT(attrib>=DP_SG_POSITION && attrib<=DP_SG_ATTR15);
        return !!(m_normalizeEnableFlags & (1<<attrib));
      }

      inline void VertexAttributeSet::setVertices( const dp::math::Vec3f * vertices, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_POSITION, 3, dp::DT_FLOAT_32, &vertices[0], 0, count, enable );
      }

      inline void VertexAttributeSet::setVertices( unsigned int pos, const dp::math::Vec3f * vertices, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_POSITION, pos, 3, dp::DT_FLOAT_32, &vertices[0], 0, count, enable );
      }

      inline Buffer::ConstIterator<dp::math::Vec3f>::Type VertexAttributeSet::getVertices() const
      {
        // debug checks on current limitations
        DP_ASSERT( getSizeOfVertexData(DP_SG_POSITION) == 3 );  
        DP_ASSERT( getTypeOfVertexData(DP_SG_POSITION) == dp::DT_FLOAT_32 );

        return getVertexData<dp::math::Vec3f>(DP_SG_POSITION);
      }

      inline unsigned int VertexAttributeSet::getNumberOfVertices() const
      {
        return getNumberOfVertexData(DP_SG_POSITION);
      }

      inline unsigned int VertexAttributeSet::getPerVertexBytes() const
      {
        unsigned int sz = 0;
        for ( AttributeContainer::const_iterator it = m_vattribs.begin() ; it != m_vattribs.end() ; ++it )
        {
          if ( isEnabled( it->first ) )
          {
            sz += it->second.getVertexDataBytes();
          }
        }
        return sz;
      }

      //inline void VertexAttributeSet::setVertexWeights( const float * weights, unsigned int count, bool enable )
      //{
      //  setVertexData( DP_VERTEX_WEIGHT, 1, dp::DT_FLOAT_32, weights, count, enable );
      //}

      //inline void VertexAttributeSet::setVertexWeights( unsigned int pos, const float * weights, unsigned int count, bool enable )
      //{
      //  setVertexData( DP_VERTEX_WEIGHT, pos, 1, dp::DT_FLOAT_32, weights, count, enable );
      //}

      //inline const void * VertexAttributeSet::getVertexWeights() const
      // {
      //  return( (     ( getSizeOfVertexData(DP_VERTEX_WEIGHT) == 1 )
      //            &&  ( getTypeOfVertexData(DP_VERTEX_WEIGHT) == dp::DT_FLOAT_32 ) )
      //          ? (const dp::math::Vec3f*)getVertexData(DP_VERTEX_WEIGHT)
      //          : NULL );
      //}

      //inline unsigned int VertexAttributeSet::getNumberOfVertexWeights() const
      //{
      //  return getNumberOfVertexData(DP_VERTEX_WEIGHT);
      //}

      inline void VertexAttributeSet::setNormals( const dp::math::Vec3f * normals, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_NORMAL, 3, dp::DT_FLOAT_32, normals, 0, count, enable );
      }

      inline void VertexAttributeSet::setNormals( unsigned int pos, const dp::math::Vec3f * normals, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_NORMAL, pos, 3, dp::DT_FLOAT_32, normals, 0, count, enable );
      }

      inline Buffer::ConstIterator<dp::math::Vec3f>::Type VertexAttributeSet::getNormals() const
      {
        // debug checks on current limitations
        DP_ASSERT( getSizeOfVertexData(DP_SG_NORMAL) == 3 );  
        DP_ASSERT( getTypeOfVertexData(DP_SG_NORMAL) == dp::DT_FLOAT_32 );

        return getVertexData<dp::math::Vec3f>(DP_SG_NORMAL);
      }

      inline unsigned int VertexAttributeSet::getNumberOfNormals() const
      {
        return getNumberOfVertexData(DP_SG_NORMAL);
      }

      template <unsigned int N, typename T> 
      inline void VertexAttributeSet::setColors( const dp::math::Vecnt<N,T> * colors, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_COLOR, N, dp::Type2EnumType<T>::type, colors, 0, count, enable );
      }

      template <unsigned int N, typename T> 
      inline void VertexAttributeSet::setColors( unsigned int pos, const dp::math::Vecnt<N,T> * colors, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_COLOR, pos, N, dp::Type2EnumType<T>::type, colors, 0, count, enable );
      }

      template <typename ValueType>
      inline typename Buffer::ConstIterator<ValueType>::Type VertexAttributeSet::getColors() const
      {
        return getVertexData<ValueType>(DP_SG_COLOR);
      }

      inline unsigned int VertexAttributeSet::getNumberOfColors() const
      {
        return getNumberOfVertexData(DP_SG_COLOR);
      }

      inline unsigned int VertexAttributeSet::getSizeOfColors() const
      {
        return getSizeOfVertexData(DP_SG_COLOR);
      }

      inline unsigned int VertexAttributeSet::getTypeOfColors() const
      {
        return getTypeOfVertexData(DP_SG_COLOR);
      }

      template <typename T> 
      inline void VertexAttributeSet::setSecondaryColors( const dp::math::Vecnt<3,T> * colors, unsigned int count, bool enable )
      {
        setVertexData(DP_SG_SECONDARY_COLOR, 3, dp::Type2EnumType<T>::type, colors, 0, count, enable );
      }

      template <typename T> 
      inline void VertexAttributeSet::setSecondaryColors( unsigned int pos, const dp::math::Vecnt<3,T> * colors, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_SECONDARY_COLOR, pos, 3, dp::Type2EnumType<T>::type, colors, 0, count, enable );
      }

      template <typename ValueType>
      inline typename Buffer::ConstIterator<ValueType>::Type VertexAttributeSet::getSecondaryColors() const
      {
        return getVertexData<ValueType>(DP_SG_SECONDARY_COLOR);
      }

      inline unsigned int VertexAttributeSet::getNumberOfSecondaryColors() const
      {
        return getNumberOfVertexData(DP_SG_SECONDARY_COLOR);
      }

      inline unsigned int VertexAttributeSet::getTypeOfSecondaryColors() const
      {
        return getTypeOfVertexData(DP_SG_SECONDARY_COLOR);
      }

      template <typename T> 
      inline void VertexAttributeSet::setFogCoords( const T * coords, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_FOG_COORD, 1, dp::Type2EnumType<T>::type, coords, 0, count, enable );
      }

      template <typename T> 
      inline void VertexAttributeSet::setFogCoords( unsigned int pos, T * coords, unsigned int count, bool enable )
      {
        setVertexData( DP_SG_FOG_COORD, pos, 1, dp::Type2EnumType<T>::type, coords, 0, count, enable );
      }

      inline Buffer::ConstIterator<float>::Type VertexAttributeSet::getFogCoords() const
      {
        return( ( getSizeOfVertexData(DP_SG_FOG_COORD) == 1 ) ? getVertexData<float>(DP_SG_FOG_COORD) : Buffer::ConstIterator<float>::Type() );
      }

      inline unsigned int VertexAttributeSet::getNumberOfFogCoords() const
      {
        return getNumberOfVertexData(DP_SG_FOG_COORD);
      }

      inline unsigned int VertexAttributeSet::getTypeOfFogCoords() const
      {
        return getTypeOfVertexData(DP_SG_FOG_COORD);
      }

      template <unsigned int N, typename T> 
      inline void VertexAttributeSet::setTexCoords( unsigned int unit, const dp::math::Vecnt<N,T> * coords, unsigned int count, bool enable )
      {
        DP_ASSERT(unit<8);
        setVertexData( DP_SG_TEXCOORD0+unit, N, dp::Type2EnumType<T>::type, coords, 0, count, enable );
      }

      template <unsigned int N, typename T> 
      inline void VertexAttributeSet::setTexCoords( unsigned int unit, unsigned int pos, const dp::math::Vecnt<N,T> * coords, unsigned int count, bool enable )
      {
        DP_ASSERT(unit<8);
        setVertexData( DP_SG_TEXCOORD0+unit, pos, N, dp::Type2EnumType<T>::type, coords, 0, count, enable );
      }

      template <typename ValueType>
      inline typename Buffer::ConstIterator<ValueType>::Type VertexAttributeSet::getTexCoords(unsigned int unit) const
      {
        DP_ASSERT(unit<8);
        return getVertexData<ValueType>(DP_SG_TEXCOORD0+unit);
      }

      inline unsigned int VertexAttributeSet::getNumberOfTexCoords(unsigned int unit) const
      {
        DP_ASSERT(unit<8);
        return getNumberOfVertexData(DP_SG_TEXCOORD0+unit);
      }

      inline unsigned int VertexAttributeSet::getSizeOfTexCoords(unsigned int unit) const
      {
        DP_ASSERT(unit<8);
        return getSizeOfVertexData(DP_SG_TEXCOORD0+unit);
      }

      inline unsigned int VertexAttributeSet::getTypeOfTexCoords(unsigned int unit) const
      {
        DP_ASSERT(unit<8);
        return getTypeOfVertexData(DP_SG_TEXCOORD0+unit);
      }

      inline bool genericAttrib(unsigned int attrib)
      {
        return attrib >= VertexAttributeSet::DP_SG_ATTR0;
      }

      inline unsigned int attribIndex(unsigned int attrib)
      {
        if ( genericAttrib(attrib) ) 
        {
          return attrib - VertexAttributeSet::DP_SG_ATTR0;
        }
        return attrib;
      }
    } // namespace core
  } // namespace sg
} // namespace dp

