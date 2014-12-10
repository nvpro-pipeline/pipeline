// Copyright NVIDIA Corporation 2011
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

#include <test/testfw/core/Test.h>
#include <test/testfw/core/Backend.h>

namespace dp
{
  namespace testfw
  {
    namespace core
    {
      //Test class for a simple one frame render
      class TestRender : public Test
      {
      public:
        DPTCORE_API TestRender();
        DPTCORE_API virtual ~TestRender();

        //Mandatory user defined function
        DPTCORE_API virtual bool onRun(unsigned int i) = 0;
        DPTCORE_API virtual bool onInit( void ) = 0;
        DPTCORE_API virtual bool onClear( void ) = 0;

        //optional
        DPTCORE_API virtual bool option( const std::vector<std::string>& optionString );

        DPTCORE_API BackendSharedPtr getBackend() const;

      public:
        DPTCORE_API virtual void render( RenderData* renderData, dp::ui::RenderTargetSharedPtr renderTarget );
        DPTCORE_API util::ImageSharedPtr getScreenshot() const;

        //This will initialize rendering back end. Every derived class MUST call this in
        //their implementation of run()
        DPTCORE_API virtual bool run( MeasurementFunctor & mf, const std::string& name );

      private:
        DPTCORE_API BackendSharedPtr createBackend( const std::string& rendererName, const std::vector<std::string>& options );

      protected:
        bool m_rendererSpecified;
        util::DynamicLibrarySharedPtr m_backendLib;
        dp::ui::RenderTargetSharedPtr m_displayTarget;
        BackendSharedPtr m_backend;

        //As long as the test launching application maintains the same renderer for each run
        //Then we might as well remember what its respective backend is to spare us the files
        //searches and queries every test launch
        static std::string m_backendName;

        unsigned int m_width;
        unsigned int m_height;
      
      };

    } // namespace core
  } // namespace testfw
} // namespace dp
