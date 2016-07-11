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

#include <dp/fx/EffectDefs.h>
#include <dp/fx/mdl/inc/MDLTokenizer.h>
#include <dp/util/FileFinder.h>
#include <set>

namespace dp
{
  namespace fx
  {
    namespace mdl
    {
      struct EnumData
      {
        EnumData()
        {}

        EnumData( std::string const& n, std::vector<std::pair<std::string,int>> const& v )
          : name(n)
          , values(v)
        {}

        std::string                             name;
        std::vector<std::pair<std::string,int>> values;
      };

      struct StageData
      {
        StageData()
        {}

        void append( StageData const& rhs )
        {
          enums.insert( rhs.enums.begin(), rhs.enums.end() );
          for ( std::vector<std::string>::const_iterator it = rhs.functions.begin(); it != rhs.functions.end(); ++it )
          {
            if ( find( functions.begin(), functions.end(), *it ) == functions.end() )
            {
              functions.push_back( *it );
            }
          }
          parameters.insert( rhs.parameters.begin(), rhs.parameters.end() );
          structures.insert( rhs.structures.begin(), rhs.structures.end() );
          temporaries.insert( rhs.temporaries.begin(), rhs.temporaries.end() );
        }

        void clear()
        {
          enums.clear();
          functions.clear();
          parameters.clear();
          structures.clear();
          temporaries.clear();
        }

        std::set<std::string>     enums;
        std::vector<std::string>  functions;    // needs to be a vector to keep dependencies in right order!
        std::set<unsigned int>    parameters;
        std::set<std::string>     structures;
        std::set<unsigned int>    temporaries;
      };

      struct SurfaceData
      {
        SurfaceData()
        {}

        std::string emission;
        std::string scattering;
      };

      struct GeometryData
      {
        GeometryData()
        {}

        std::string cutoutOpacity;
        std::string displacement;
        std::string normal;
      };

      struct TemporaryData
      {
        TemporaryData()
        {}

        TemporaryData( std::string const& t, std::string const& e )
          : type( t )
          , eval( e )
        {}

        std::string type;
        std::string eval;
        StageData   stage;
      };

      struct ParameterData
      {
        ParameterData()
        {}

        ParameterData( std::string const& n )
          : name( n )
        {}

        ParameterData( std::string const & t, std::string const & n, std::string const& v, std::string const& s = "", std::string const& a = "" )
          : type( t )
          , name( n )
          , value( v )
          , semantic( s )
          , annotations( a )
        {}

        std::string type;
        std::string name;
        std::string value;
        std::string semantic;
        std::string annotations;
      };

      struct StructureData
      {
        std::string                                     name;
        std::vector<std::pair<std::string,std::string>> members;    // pairs of type and name
      };

      struct MaterialData
      {
        MaterialData()
          : maxTemporaryIndex(~0)
          , transparent( false )
        {}

        std::vector<ParameterData>            parameters;

        std::map<unsigned int, TemporaryData> temporaries;

        std::string                           thinWalled;
        SurfaceData                           surfaceData;
        SurfaceData                           backfaceData;
        std::string                           ior;
        //VolumeData                            volumeData;   No volume data gathered!
        GeometryData                          geometryData;

        std::map<std::string,EnumData>        enums;
        unsigned int                          maxTemporaryIndex;
        std::vector<unsigned int>             parameterIndirection;
        std::map<dp::fx::Domain, StageData>   stageData;
        std::map<std::string,StructureData>   structures;
        bool                                  transparent;
        std::set<std::string>                 varyings;
      };

      struct FunctionData
      {
        FunctionData()
        {}

        std::set<std::string>       functionDependencies;
        std::set<std::string>       varyingDependencies;
      };


      class MaterialBuilder : public dp::fx::mdl::MDLTokenizer
      {
        public:
          MaterialBuilder( std::string const& configFile );
          std::map<std::string,MaterialData> const& getMaterials() const;

        protected:
          bool annotationBegin( std::string const& name );
          void annotationEnd();
          bool argumentBegin( unsigned int idx, std::string const& type, std::string const& name );
          void argumentEnd();
          bool arrayBegin( std::string const& type, size_t size );
          void arrayEnd();
          bool callBegin( std::string const& type, std::string const& name );
          void callEnd();
          void defaultRef( std::string const& type );
          bool enumTypeBegin( std::string const& name, size_t size );
          void enumTypeEnd();
          void enumTypeValue( std::string const& name, int value );
          bool fieldBegin( std::string const& name );
          void fieldEnd();
          bool fileBegin( std::string const& name );
          void fileEnd();
          bool materialBegin( std::string const& name, dp::math::Vec4ui const& hash );
          void materialEnd();
          bool matrixBegin( std::string const& type );
          void matrixEnd();
          bool parameterBegin( unsigned int index, std::string const& name );
          void parameterEnd();
          void referenceParameter( unsigned int idx );
          void referenceTemporary( unsigned int idx );
          bool structureBegin( std::string const& type );
          void structureEnd();
          bool structureTypeBegin( std::string const& name );
          void structureTypeElement( std::string const& type, std::string const& name );
          void structureTypeEnd();
          bool temporaryBegin( unsigned int idx );
          void temporaryEnd();
          void valueBool( bool value );
          void valueBsdfMeasurement( std::string const& value );
          void valueColor( dp::math::Vec3f const& value );
          void valueEnum( std::string const& type, int value, std::string const& name );
          void valueFloat( float value );
          void valueInt( int value );
          void valueLightProfile( std::string const& value );
          void valueString( std::string const& value );
          void valueTexture( std::string const& name, GammaMode gamma );
          bool vectorBegin( std::string const& type );
          void vectorEnd();

        private:
          typedef struct Argument
          {
            Argument()
            {}

            Argument( std::string const& n )
              : name( n )
            {}

            void clear()
            {
              name.clear();
              arguments.clear();
            }

            bool empty() const
            {
              return( name.empty() && arguments.empty() );
            }

            std::string                                   name;
            std::vector<std::pair<std::string, Argument>> arguments;
          };

        private:
          void getSurfaceData( SurfaceData & surfaceData );
          Argument & getTargetArg( Argument & arg, size_t idx ) const;
          std::string resolveAnnotations();
          std::string resolveArgument( Argument & arg, bool embrace = false );
          void storeFunctionCall( std::string const& name );
          std::string translateType( std::string const& type );

        private:
          std::vector<Argument>                         m_annotations;
          Argument                                      m_argument;
          std::stack<Argument*>                         m_currentCall;
          std::map<std::string,EnumData>::iterator      m_currentEnum;
          std::map<std::string,MaterialData>::iterator  m_currentMaterial;
          StageData *                                   m_currentStage;
          unsigned int                                  m_currentTemporaryIdx;
          dp::util::FileFinder                          m_fileFinder;
          std::map<std::string,FunctionData>            m_functions;
          bool                                          m_insideAnnotation;
          bool                                          m_insideParameter;
          std::map<std::string,MaterialData>            m_materials;
          std::stack<StructureData>                     m_structureStack;
          std::map<unsigned int,unsigned int>           m_temporaryBuddies;
          std::map<unsigned int,std::string>            m_temporarySamplerMap;
          StageData                                     m_temporaryStage;
      };

    } // mdl
  } // fx
} // dp
