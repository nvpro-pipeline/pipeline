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


#if defined(HAVE_NVPMAPI)

#include <dp/util/Config.h>

#define NVPM_INITGUID   // needs to be defined to get the ETID_NvPmApi !!

#include <iostream>
#include <iomanip>
#include <fstream>
#include <dp/util/Array.h>
#include <dp/util/NVPerfMon.h>

using std::make_pair;
using std::pair;
using std::string;
using std::vector;

namespace dp
{
  namespace util
  {
    NVPerfMon::NVPerfMon( const vector<string> & counterFilter )
      : m_nvPmApi(nullptr)
      , m_nvPmApiContext(0)
      , m_numPasses(0)
      , m_pass(0)
      , m_counterFilter(counterFilter)
      , m_durationCounterID(0)
      , m_deviationCounterID(0)
    {
    }

    NVPerfMon::~NVPerfMon()
    {
      if ( m_nvPmApi )
      {
        m_nvPmApi->DestroyContext( m_nvPmApiContext );
        m_nvPmApi->Shutdown();
      }
    }

    NVPerfMon * NVPerfMon::m_nvPerfMon = nullptr;

    int NVPerfMon::nvPmApiEnumFunction( NVPMCounterID counterID, const char *pcCounterName )
    {
      DP_ASSERT( m_nvPerfMon );
      DP_ASSERT( m_nvPerfMon->m_counterIDToSpec.find( counterID ) == m_nvPerfMon->m_counterIDToSpec.end() );
      DP_ASSERT( m_nvPerfMon->m_nameToCounterID.find( pcCounterName ) == m_nvPerfMon->m_nameToCounterID.end() );

      char description[4096];
      NVPMUINT length = sizeof(description);
      m_nvPerfMon->m_nvPmApi->GetCounterDescription( counterID, description, &length );

      NVPMUINT64 counterValue;
      DP_VERIFY( m_nvPerfMon->m_nvPmApi->GetCounterAttribute( counterID, NVPMA_COUNTER_VALUE, &counterValue ) == NVPM_OK );

      CounterSpec spec;
      spec.name = pcCounterName;
      spec.description = description;
      spec.percentage = ( counterValue == NVPM_CV_PERCENT );

      m_nvPerfMon->m_counterIDToSpec.insert( make_pair( counterID, spec ) );
      m_nvPerfMon->m_nameToCounterID.insert( make_pair( pcCounterName, counterID ) );

      return( NVPM_OK );
    }

    bool NVPerfMon::init()
    {
      if ( !m_nvpmLib )
      {
        m_nvpmLib = dp::util::DynamicLibrary::createFromFile( "NvPmApi.Core.dll" );
        if ( m_nvpmLib )
        {
          NVPMGetExportTable_Pfn nvpmGetExportTable = (NVPMGetExportTable_Pfn)m_nvpmLib->getSymbol( "NVPMGetExportTable" );
          DP_ASSERT( nvpmGetExportTable );
          DP_VERIFY( nvpmGetExportTable( &ETID_NvPmApi, (void**) &m_nvPmApi ) == NVPM_OK );
          DP_VERIFY( m_nvPmApi->Init() == NVPM_OK );

          HGLRC currentContext = wglGetCurrentContext();
          DP_VERIFY( m_nvPmApi->CreateContextFromOGLContext( (APIContextHandle)currentContext, &m_nvPmApiContext ) == NVPM_OK );
          m_nvPerfMon = this;
          DP_VERIFY( m_nvPmApi->EnumCountersByContext( m_nvPmApiContext, &nvPmApiEnumFunction ) == NVPM_OK );

#if 0
          // spit out all the counters
          std::ofstream of( "PerfMonDescriptions.txt" );
          for ( CounterIDToSpecMap::const_iterator it = m_counterIDToSpec.begin() ; it != m_counterIDToSpec.end() ; ++it )
          {
            of << std::setw(44) << std::left << it->second.name << " : " << it->second.description << std::endl;
          }
#endif

          if ( m_counterFilter.empty() )
          {
            for ( CounterIDToSpecMap::const_iterator it = m_counterIDToSpec.begin() ; it != m_counterIDToSpec.end() ; ++it )
            {
              DP_VERIFY( m_nvPmApi->AddCounter( m_nvPmApiContext, it->first ) == NVPM_OK );
              m_counterIDs.push_back( it->first );
            }
          }
          else
          {
            for ( CounterIDToSpecMap::const_iterator it = m_counterIDToSpec.begin() ; it != m_counterIDToSpec.end() ; ++it )
            {
              for ( size_t i=0 ; i<m_counterFilter.size() ; i++ )
              {
                if ( it->second.name.find( m_counterFilter[i] ) != string::npos )
                {
                  DP_VERIFY( m_nvPmApi->AddCounter( m_nvPmApiContext, it->first ) == NVPM_OK );
                  m_counterIDs.push_back( it->first );
                }
              }
            }
          }
          for ( size_t i=0 ; i<m_counterIDs.size() ; i++ )
          {
            // "Bottleneck" and "SOL" additionally hold the approximate time spend in the shader in their "cycles" value
            // In endExperiment, I use those values to calculate a duration and a deviation, and add them to the container of results
            if ( ( m_counterIDToSpec[m_counterIDs[i]].name.find( "Bottleneck" ) != string::npos )
              || ( m_counterIDToSpec[m_counterIDs[i]].name.find( "SOL" ) != string::npos ) )
            {
              // add two counters, we derive from others
              DP_ASSERT( !m_counterIDToSpec.empty() );
              m_durationCounterID = m_counterIDToSpec.rbegin()->first + 1;
              CounterSpec durationSpec;
              durationSpec.name = "Derived Duration";
              durationSpec.description = "Average Duration in Picoseconds";
              durationSpec.percentage = false;
              m_counterIDToSpec.insert( make_pair( m_durationCounterID, durationSpec ) );

              m_deviationCounterID = m_durationCounterID + 1;
              CounterSpec deviationSpec;
              deviationSpec.name = "Derived Deviation";
              deviationSpec.description = "Deviation from Average Duration (%)";
              deviationSpec.percentage = true;
              m_counterIDToSpec.insert( make_pair( m_deviationCounterID, deviationSpec ) );

              m_counterIDs.push_back( m_durationCounterID );
              m_counterIDs.push_back( m_deviationCounterID );
              break;
            }
          }
          m_samples.resize( m_counterIDs.size() );
        }
      }

      return( m_nvpmLib );
    }

    void NVPerfMon::beginExperiment()
    {
      DP_ASSERT( m_nvPmApi );
      DP_VERIFY( m_nvPmApi->BeginExperiment( m_nvPmApiContext, &m_numPasses ) == NVPM_OK );
      m_pass = 0;
    }

    bool NVPerfMon::endExperiment( std::vector<CounterResult> & results )
    {
      DP_ASSERT( m_nvPmApi );
      DP_VERIFY( m_nvPmApi->EndExperiment( m_nvPmApiContext ) == NVPM_OK );

      NVPMUINT numSamples = checked_cast<NVPMUINT>(m_samples.size());
      if ( numSamples )
      {
        results.resize( numSamples );
        vector<NVPMUINT64> cycleBuffer;
        for ( size_t idx=0 ; idx<m_counterIDs.size() ; idx++ )
        {
          NVPMUINT64 value, cycles;
          if ( m_nvPmApi->GetCounterValue( m_nvPmApiContext, m_counterIDs[idx], 0, &value, &cycles ) == NVPM_OK )
          {
            if ( m_counterIDToSpec[m_counterIDs[idx]].percentage )
            {
              results[idx].percent = 100.0 * double(value) / double(cycles);
            }
            else
            {
              results[idx].count = value;
            }
            // get the cycles for the "Bottleneck" and "SOL" counters, which are the approximate time
            // in picoseconds spend in the shader
            if ( ( m_counterIDToSpec[m_counterIDs[idx]].name.find( "Bottleneck" ) != string::npos )
              || ( m_counterIDToSpec[m_counterIDs[idx]].name.find( "SOL" ) != string::npos ) )
            {
              cycleBuffer.push_back( cycles );
            }
          }
        }
        if ( ! cycleBuffer.empty() )
        {
          // use the average of the cycles as a complexity measure
          NVPMUINT64 average = 0;
          for ( size_t i=0 ; i<cycleBuffer.size() ; i++ )
          {
            average += cycleBuffer[i];
          }
          average /= cycleBuffer.size();
          results[results.size()-2].count = average;    // the second to last entry is the duration

          // determine the maximal deviation from that average in percent
          NVPMUINT64 maxDev = 0;
          for ( size_t i=0 ; i<cycleBuffer.size() ; i++ )
          {
            NVPMUINT64 dev = ( average < cycleBuffer[i] ) ? ( cycleBuffer[i] - average ) : ( average - cycleBuffer[i] );
            if ( maxDev < dev )
            {
              maxDev = dev;
            }
          }
          results[results.size()-1].percent = 100.0 * double(maxDev) / double(average);
        }
      }
      return( !!numSamples );
    }

    void NVPerfMon::beginPass()
    {
      DP_ASSERT( m_nvPmApi );
      DP_ASSERT( m_pass < m_numPasses );
      DP_VERIFY( m_nvPmApi->BeginPass( m_nvPmApiContext, m_pass ) == NVPM_OK );
      DP_VERIFY( m_nvPmApi->BeginObject( m_nvPmApiContext, 0 ) == NVPM_OK );
    }

    void NVPerfMon::endPass()
    {
      DP_ASSERT( m_nvPmApi );
      DP_ASSERT( m_pass < m_numPasses );
      DP_VERIFY( m_nvPmApi->EndObject( m_nvPmApiContext, 0 ) == NVPM_OK );
      DP_VERIFY( m_nvPmApi->EndPass( m_nvPmApiContext, m_pass++ ) == NVPM_OK );
    }

    bool NVPerfMon::finishedExperiment() const
    {
      return( m_numPasses <= m_pass );
    }

    const vector<NVPMCounterID> & NVPerfMon::getCounterIDs() const
    {
      return( m_counterIDs );
    }

    const NVPerfMon::CounterIDToSpecMap & NVPerfMon::getCounterIDToSpecMap() const
    {
      return( m_counterIDToSpec );
    }

  } // namespace util
} // namespace dp

// HAVE_NVPMAPI
#endif
