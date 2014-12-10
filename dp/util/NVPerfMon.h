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
/** \file */

#if defined(HAVE_NVPMAPI)

#include <NvPmApi.h>
#include <dp/util/Config.h>
#include <dp/util/DynamicLibrary.h>

namespace dp
{
  namespace util
  {
    class NVPerfMon
    {
      public:
        typedef struct
        {
          std::string name;
          std::string description;
          bool        percentage;
        }                                           CounterSpec;
        typedef std::map<NVPMCounterID,CounterSpec> CounterIDToSpecMap;
        typedef std::map<std::string,NVPMCounterID> NameToCounterIDMap;
        typedef union
        {
          NVPMUINT64    count;
          double        percent;
        }                                           CounterResult;

      public:
        DP_UTIL_API NVPerfMon( const std::vector<std::string> & counterFilter = std::vector<std::string>() );
        DP_UTIL_API ~NVPerfMon();

        DP_UTIL_API bool init();
        DP_UTIL_API void beginExperiment();
        DP_UTIL_API bool endExperiment( std::vector<CounterResult> & results );
        DP_UTIL_API void beginPass();
        DP_UTIL_API void endPass();
        DP_UTIL_API bool finishedExperiment() const;

        DP_UTIL_API const std::vector<NVPMCounterID> & getCounterIDs() const;
        DP_UTIL_API const CounterIDToSpecMap & getCounterIDToSpecMap() const;

      private:
        static int nvPmApiEnumFunction( NVPMCounterID unCounterID, const char *pcCounterName );

      private:
        static NVPerfMon * m_nvPerfMon;

      private:
        dp::util::DynamicLibrarySharedPtr   m_nvpmLib;
        NvPmApi                           * m_nvPmApi;
        NVPMContext                         m_nvPmApiContext;
        CounterIDToSpecMap                  m_counterIDToSpec;
        NameToCounterIDMap                  m_nameToCounterID;
        std::vector<NVPMSampleValue>        m_samples;
        NVPMUINT                            m_numPasses;
        NVPMUINT                            m_pass;
        std::vector<std::string>            m_counterFilter;
        std::vector<NVPMCounterID>          m_counterIDs;
        NVPMCounterID                       m_durationCounterID;
        NVPMCounterID                       m_deviationCounterID;
    };
  } // namespace util
} // namespace dp

// HAVE_NVPMAPI
#endif
