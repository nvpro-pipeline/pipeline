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
/** \file */

#include <dp/util/Config.h>

#include <vector>
#include <string>
#include <sstream>

#include <stdio.h>
#include <iomanip>

namespace dp
{
  namespace util
  {
    //! Separate a string into tokens
    /** Consecutive calls to getNextToken() return the tokens of \a inputString
        that are separated by one or multiple occurences of characters from \a delimString.
        If no more tokens are left, an empty string will be returned and hasMoreTokens() will return false.
      */
    class StrTokenizer
    {
      public:
        /*! \brief Create a StrTokenizer with a specified delimiter string.
         *  \param delimString String holding the delimiter characters. */
        DP_UTIL_API explicit StrTokenizer( const std::string& delimString );

        /*! \brief Destructor of a StrTokenizer. */
        DP_UTIL_API ~StrTokenizer();

        /*! \brief Set the string to tokenize
         *  \param inputString The string to tokenize. */
        DP_UTIL_API void setInput( const std::string& inputString );

        /*! \brief Get the next token out of the string to tokenize.
         *  \return A string with the next token in the input string. */
        DP_UTIL_API const std::string& getNextToken();

        /*! \brief Check if the input string is completely tokenized.
         *  \return \c true, if there are more tokens in the input string, otherwise \c false. */
        DP_UTIL_API bool hasMoreTokens() const;

      private:
        unsigned int m_currentPos;
        std::string m_inputString;
        std::string m_token;
        std::string m_delim;
    };

  } // namespace util
} // namespace dp


