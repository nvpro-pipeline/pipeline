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

#include <dp/fx/Config.h>
#include <dp/fx/EffectData.h>
#include <dp/fx/EffectDefs.h>
#include <dp/fx/EffectSpec.h>
#include <dp/fx/EffectLibrary.h>
#include <dp/fx/inc/Snippet.h>
#include <dp/util/SharedPtr.h>
#include <string>

namespace dp
{
  namespace fx
  {
    class EffectLibraryImpl;

    SMART_TYPES( EffectLoader );

    class EffectLoader
    {
    public:
      DP_FX_API virtual ~EffectLoader();

      DP_FX_API virtual bool loadEffects( const std::string & filename ) = 0;
      DP_FX_API virtual bool save( const SmartEffectData& effectData, const std::string& filename ) = 0;

      DP_FX_API virtual SmartShaderPipeline generateShaderPipeline( const dp::fx::ShaderPipelineConfiguration& configuration ) = 0;
      DP_FX_API virtual bool effectHasTechnique( SmartEffectSpec const& effectSpec, std::string const& techniqueName, bool rasterizer ) = 0;

    protected:
      DP_FX_API EffectLoader( EffectLibraryImpl * effectLibrary );
      EffectLibraryImpl * getEffectLibrary() const;

    private:
      EffectLibraryImpl * m_effectLibrary;
    };


    inline EffectLoader::EffectLoader( EffectLibraryImpl * effectLibrary )
      : m_effectLibrary( effectLibrary )
    {
      DP_ASSERT( m_effectLibrary );
    }

    inline EffectLoader::~EffectLoader()
    {
    }

    inline EffectLibraryImpl * EffectLoader::getEffectLibrary() const
    {
      return( m_effectLibrary );
    }

  } // namespace fx
} // namespace dp