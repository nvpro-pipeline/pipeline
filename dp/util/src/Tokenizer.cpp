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


#include <dp/util/DPAssert.h>
#include <dp/util/Config.h>
#include <dp/util/Tokenizer.h>

namespace dp
{
  namespace util
  {

    //////////////////////////////////////////////////////////////////////////
    // String Tokenizer

    StrTokenizer::StrTokenizer( std::string const& delim )
    : m_currentPos( 0 )
    {
      m_delim = delim;
    }

    StrTokenizer::~StrTokenizer()
    {
    }

    void StrTokenizer::setInput( std::string const& inputString )
    {
      m_inputString = inputString;
      m_currentPos = 0;
    }

    bool StrTokenizer::hasMoreTokens() const
    {
      return m_inputString.find_first_not_of(m_delim, m_currentPos) != std::string::npos;  
    }

    std::string const& StrTokenizer::getNextToken()
    {
      std::string::size_type first, last;

      // find beginning of first token starting at m_currentPos
      first = m_inputString.find_first_not_of(m_delim, m_currentPos);
      if ( first != std::string::npos )
      {
        // find end of current token
        last = m_inputString.find_first_of(m_delim, first);
        if ( last == std::string::npos ) 
        { 
          // end of m_inputString if no delimiter found
          last = m_inputString.length();
        }
        // copy the found token and update m_currentPos
        m_token = m_inputString.substr(first, last-first);
        m_currentPos = (unsigned int)(last);
      }
      else
      {
        // no more tokens in m_inputString
        m_token = "";
        m_currentPos = (unsigned int)(m_inputString.length());
      }
      return m_token;
    }

  } // namespace util
} // namespace dp

