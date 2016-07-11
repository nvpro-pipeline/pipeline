// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/fx/mdl/Config.h>
#include <dp/math/Matmnt.h>
#include <dp/util/DynamicLibrary.h>
#include <dp/util/FileFinder.h>
#include <dp/util/Timer.h>
#include <mi/mdl_sdk.h>
#include <map>
#include <stack>
#include <vector>

namespace dp
{
  namespace fx
  {
    namespace mdl
    {
      class MDLTokenizer
      {
        public:
          DP_FX_MDL_API MDLTokenizer();
          DP_FX_MDL_API ~MDLTokenizer();

          DP_FX_MDL_API void parseFile( std::string const& file, dp::util::FileFinder const& fileFinder );
          DP_FX_MDL_API void setFilterDefaults( bool filter );

          enum class GammaMode
          {
            DEFAULT,
            LINEAR,
            SRGB
          };

        protected:
          virtual bool annotationBegin( std::string const& name ) = 0;
          virtual void annotationEnd() = 0;
          virtual bool argumentBegin( unsigned int idx, std::string const& type, std::string const& name ) = 0;
          virtual void argumentEnd() = 0;
          virtual bool arrayBegin( std::string const & type, size_t size ) = 0;
          virtual void arrayEnd() = 0;
          virtual bool callBegin( std::string const& type, std::string const& name ) = 0;
          virtual void callEnd() = 0;
          virtual void defaultRef( std::string const& type ) = 0;
          virtual bool enumTypeBegin( std::string const& name, size_t size ) = 0;
          virtual void enumTypeEnd() = 0;
          virtual void enumTypeValue( std::string const& name, int value ) = 0;
          virtual bool fieldBegin( std::string const& name ) = 0;
          virtual void fieldEnd() = 0;
          virtual bool fileBegin( std::string const& name ) = 0;
          virtual void fileEnd() = 0;
          virtual bool materialBegin( std::string const& name, dp::math::Vec4ui const& hash ) = 0;
          virtual void materialEnd() = 0;
          virtual bool matrixBegin( std::string const& type ) = 0;
          virtual void matrixEnd() = 0;
          virtual bool parameterBegin( unsigned int index, std::string const& name ) = 0;
          virtual void parameterEnd() = 0;
          virtual void referenceParameter( unsigned int idx ) = 0;
          virtual void referenceTemporary( unsigned int idx ) = 0;
          virtual bool structureBegin( std::string const& type ) = 0;
          virtual void structureEnd() = 0;
          virtual bool structureTypeBegin( std::string const& name ) = 0;
          virtual void structureTypeElement( std::string const& type, std::string const& name ) = 0;
          virtual void structureTypeEnd() = 0;
          virtual bool temporaryBegin( unsigned int idx ) = 0;
          virtual void temporaryEnd() = 0;
          virtual void valueBool( bool value ) = 0;
          virtual void valueBsdfMeasurement( std::string const& value ) = 0;
          virtual void valueColor( dp::math::Vec3f const& value ) = 0;
          virtual void valueEnum( std::string const& type, int value, std::string const& name ) = 0;
          virtual void valueFloat( float value ) = 0;
          virtual void valueInt( int value ) = 0;
          virtual void valueLightProfile( std::string const& value ) = 0;
          virtual void valueString( std::string const& value ) = 0;
          virtual void valueTexture( std::string const& file, GammaMode gamma ) = 0;
          virtual bool vectorBegin( std::string const& type ) = 0;
          virtual void vectorEnd() = 0;

          void triggerTokenizeFunctionReturnType( std::string functionName );

        private:
          DP_FX_MDL_API bool checkDefaultField(std::string const& fieldName, mi::base::Handle<mi::neuraylib::IExpression const> const& expression);
          DP_FX_MDL_API void tokenizeArgument(mi::Size idx, std::string const& name, mi::base::Handle<mi::neuraylib::IExpression const> const& argumentExpression, mi::base::Handle<mi::neuraylib::IExpression const> const& defaultExpression);
          DP_FX_MDL_API void tokenizeArray( mi::base::Handle<mi::neuraylib::IValue_array const> const& value );
          DP_FX_MDL_API void tokenizeBSDFMeasurement( mi::base::Handle<mi::neuraylib::IValue_bsdf_measurement const> const& value );
          DP_FX_MDL_API void tokenizeColor( mi::base::Handle<mi::neuraylib::IValue_color const> const& value );
          DP_FX_MDL_API void tokenizeDirectCall( mi::base::Handle<mi::neuraylib::IExpression_direct_call const> const& call );
          DP_FX_MDL_API void tokenizeEnum( mi::base::Handle<mi::neuraylib::IValue_enum const> const& value );
          DP_FX_MDL_API void tokenizeEnumType( mi::base::Handle<mi::neuraylib::IType_enum const> const& type );
          DP_FX_MDL_API void tokenizeExpression( mi::base::Handle<mi::neuraylib::IExpression const> const& expression );
          DP_FX_MDL_API void tokenizeField( std::string const& fieldName );
          DP_FX_MDL_API void tokenizeLightProfile( mi::base::Handle<mi::neuraylib::IValue_light_profile const> const& value );
          DP_FX_MDL_API void tokenizeMaterial( std::string const& name );
          DP_FX_MDL_API void tokenizeMatrix( mi::base::Handle<mi::neuraylib::IValue_matrix const> const& value );
          DP_FX_MDL_API void tokenizeParameter( mi::Size parameterIndex );
          DP_FX_MDL_API void tokenizeParameterExpression( mi::base::Handle<mi::neuraylib::IExpression_parameter const> const& expression );
          DP_FX_MDL_API void tokenizeStructure( mi::base::Handle<mi::neuraylib::IValue_struct const> const& value );
          DP_FX_MDL_API void tokenizeStructureType( mi::base::Handle<mi::neuraylib::IType_struct const> const& type );
          DP_FX_MDL_API void tokenizeTemporary( mi::Size temporaryIndex );
          DP_FX_MDL_API void tokenizeTexture( mi::base::Handle<mi::neuraylib::IValue_texture const> const& value );
          DP_FX_MDL_API void tokenizeType( mi::base::Handle<mi::neuraylib::IType const> const& type );
          DP_FX_MDL_API void tokenizeValue( mi::base::Handle<mi::neuraylib::IValue const> const& value );
          DP_FX_MDL_API void tokenizeVector( mi::base::Handle<mi::neuraylib::IValue_vector const> const& value );

        private:
          mi::base::Handle<mi::neuraylib::ICompiled_material>         m_compiledMaterial;
          mi::base::Handle<mi::neuraylib::IDatabase>                  m_database;
          bool                                                        m_filterDefaults;
          mi::base::Handle<mi::neuraylib::IMaterial_definition const> m_materialDefinition;
          mi::base::Handle<mi::neuraylib::IMdl_compiler>              m_mdlCompiler;
          mi::base::Handle<mi::neuraylib::IExpression_factory>        m_mdlExpressionFactory;
          mi::base::Handle<mi::neuraylib::IMdl_factory>               m_mdlFactory;
          dp::util::DynamicLibrarySharedPtr                           m_mdlSDK;
          mi::base::Handle<mi::neuraylib::IValue_factory>             m_mdlValueFactory;
          mi::base::Handle<mi::neuraylib::INeuray>                    m_neuray;
          mi::base::Handle<mi::neuraylib::ITransaction>               m_transaction;
      };

    } // mdl
  } // fx
} //dp
