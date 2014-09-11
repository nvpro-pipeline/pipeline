// Copyright NVIDIA Corporation 2012
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

namespace dp
{
  namespace rix
  {
    namespace gl
    {
      /************************************************************************/
      /* ParameterObject                                                      */
      /************************************************************************/
      class ParameterObject : public dp::rix::core::HandledObject
      {
      public:

        typedef void (ParameterObject::*ConversionFunction)( void const* convertedData) const;

        ParameterObject( unsigned int offset, size_t convertedSize, ConversionFunction conversionFunction )
          : m_conversionFunction( conversionFunction )
          , m_convertedSize( convertedSize )
          , m_offset( offset )
        {
        }

        virtual void update( const void* data ) = 0;

        /** \brief copy and convert to the required datatype **/
        virtual void copy( const void* containerData, void* destination ) const = 0;

        /** \brief query #bytes required for the converted data **/

        size_t getConvertedSize() const;

        /** \brief update parameter from converted data. **/
        void updateConverted( void const* convertedData ) const;

      protected:
        ConversionFunction m_conversionFunction;
        size_t m_convertedSize;
        unsigned int m_offset;
      };

      inline size_t ParameterObject::getConvertedSize() const
      {
        return m_convertedSize;
      }

      inline void ParameterObject::updateConverted( void const* convertedData ) const
      {
        (this->*m_conversionFunction)( convertedData );
      }

      typedef dp::rix::core::SmartHandle<ParameterObject> SmartParameterObject;

      /************************************************************************/
      /* ParameterDummy                                                       */
      /************************************************************************/
      class ParameterDummy : public ParameterObject
      {
      public:
        ParameterDummy()
          : ParameterObject( ~0, 0, static_cast<ConversionFunction>(&ParameterDummy::updateConverted) )
        {}
        // TODO: warn somewhere when this class is instantiated
        // as it resembles a parameter that is in a descriptor,
        // but not in the shader
        virtual void update( const void* /*data*/ ) { };
        virtual void copy( const void* /*containerData*/, void* /*destination*/ ) const {};
        void updateConverted( void const * /*converted*/ ) const {};
      };

#if 0
      // Those are not yet supported by the new engine!
      /************************************************************************/
      /* AttributeParameternt<n,T>                                            */
      /************************************************************************/
      template<unsigned int n, typename T>
      class AttributeParameternt : public ParameterObject
      {
      public:
        AttributeParameternt( unsigned int offset, unsigned int attributeIndex );
        virtual void update( const void* data );
        virtual void copy( const void* containerData, void* destination ) const;
        void doUpdateConverted( void const* converted ) const;

        unsigned int m_attributeIndex;
      };

      /************************************************************************/
      /* AttributeParameterBufferAddress                                      */
      /************************************************************************/
      class AttributeParameterBufferAddress : public ParameterObject
      {
      public:
        AttributeParameterBufferAddress( unsigned int offset, unsigned int attribute, unsigned int arraySize );
        virtual void update( const void* data );
        virtual void copy( const void* containerData, void* destination ) const;
        void doUpdateConverted( void const* converted ) const;

        unsigned int m_attribute;
        unsigned int m_arraySize;
      };
#endif


    } // namespace gl
  } // namespace rix
} // namespace dp
