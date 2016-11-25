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

#include <map>
#include <tinyxml.h>
#include <dp/fx/mdl/inc/MaterialBuilder.h>

class GLSLBuilder
{
  public:
    TiXmlElement * buildPipelines( std::map<std::string,dp::fx::mdl::MaterialData> const& materials );

  private :
    struct VaryingData
    {
      VaryingData()
      {}

      VaryingData( std::string const & t, std::string const & e )
        : type(t)
        , eval(e)
      {}

      std::string type;
      std::string eval;
    };

  private:
    void buildAttributes( TiXmlElement * glslElement );
    void buildEffect( TiXmlElement * parent, dp::fx::Domain domain, std::map<std::string,dp::fx::mdl::MaterialData>::const_iterator material );
    void buildEnums( TiXmlElement * parent, std::map<std::string,dp::fx::mdl::MaterialData> const& materials );
    void buildParameter( TiXmlElement * parent, dp::fx::mdl::ParameterData const& pd );
    void buildParameterGroup( TiXmlElement * parent, std::set<unsigned int> const& stageParameters, std::vector<std::pair<size_t,size_t>> const& materialParameters, std::vector<dp::fx::mdl::ParameterData> const& parameterData, std::string const& materialName );
    void buildPipelineSpec( TiXmlElement * parent, std::string const& baseName );
    void buildSourceElement( TiXmlElement * parent, std::string const& source );
    void buildSourceElement( TiXmlElement * parent, std::string const& input, std::string const& name, std::string const& location );
    void buildSourceElementEnums( TiXmlElement * parent, std::set<std::string> const& usedEnums, std::map<std::string,dp::fx::mdl::EnumData> const& enumData );
    void buildSourceElementEvalIOR( TiXmlElement * parent, std::string const& ior );
    void buildSourceElementEvalSurface( TiXmlElement * parent, dp::fx::mdl::SurfaceData const& surfaceData, std::string const& postFix );
    void buildSourceElementEvalTemporaries(TiXmlElement * parent, std::set<unsigned int> const& stageTemporaries, std::map<unsigned int, dp::fx::mdl::TemporaryData> const& temporaries, dp::fx::Domain domain);
    void buildSourceElementEvalThinWalled(TiXmlElement * parent, std::string const& thinWalled);
    void buildSourceElementEvalVaryings( TiXmlElement * parent, std::set<std::string> const& varyings );
    void buildSourceElementFunctions( TiXmlElement * parent, std::vector<std::string> const& functions );
    void buildSourceElementEvalGeometry( TiXmlElement * parent, dp::fx::mdl::GeometryData const& geometryData, dp::fx::Domain domain );
    void buildSourceElementGlobals( TiXmlElement * parent, dp::fx::Domain domain );
    void buildSourceElementStructures( TiXmlElement * parent, std::set<std::string> const& structures );
    void buildSourceElementTemporaries(TiXmlElement * parent, std::set<unsigned int> const& temporariesSet, std::map<unsigned int, dp::fx::mdl::TemporaryData> const& temporariesMap);
    void buildSourceElementVaryings( TiXmlElement * parent, std::set<std::string> const& varyings, std::string const& prefix );
    void buildTechniqueDepthPass( TiXmlElement * parent, std::map<dp::fx::Domain,dp::fx::mdl::StageData>::const_iterator stage, dp::fx::mdl::MaterialData const& material );
    void buildTechniqueForward( TiXmlElement * parent, std::map<dp::fx::Domain,dp::fx::mdl::StageData>::const_iterator stage, dp::fx::mdl::MaterialData const& material );
    VaryingData const& getVaryingData( std::string const& varyingName );

  private:
    std::map<std::string,dp::fx::mdl::StructureData> m_structures;
};
