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


#include <test/testfw/core/MeasurementFunctorGoldImage.h>
#include <dp/util/File.h>
#include <dp/util/DPAssert.h>
#include <iostream>

#include <boost/program_options.hpp>

namespace options = boost::program_options;

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      MeasurementFunctorGoldImage::MeasurementFunctorGoldImage()
        : m_goldDir( util::getCurrentPath() + "\\dpt\\image" )
        , m_imageDir( util::getCurrentPath() + "\\dpt\\gold" )
        , m_gold(false)
        , m_filenamePrefix("")
        , m_filenameSuffix("")
      {
      }

      bool MeasurementFunctorGoldImage::option( const std::vector<std::string>& optionString )
      {
        options::options_description od("Usage: MeasurementFunctorGoldImage");
        od.add_options()
          ( "gold", "Whether or not to generate gold images" )
          ( "goldDir", options::value<std::string>(), "Directory to dump gold images to" )
          ( "imageDir", options::value<std::string>(), "Directory to dump the screenshots to" )
          ( "outputFilenamePrefix", options::value<std::string>(), "Prefix to add to outputted file (Optional)" )
          ( "outputFilenameSuffix", options::value<std::string>(), "Suffix to add to outputted file (Optional)" )
          ;

        options::basic_parsed_options<char> parsedOpts = options::basic_command_line_parser<char>( optionString ).options( od ).allow_unregistered().run();

        options::variables_map optsMap;
        options::store( parsedOpts, optsMap );

        // Gold

        m_gold = !optsMap["gold"].empty();

        // Gold Dir
        if( optsMap["goldDir"].empty() )
        {
          std::cerr << "Warning: No gold directory was specified\n";
        }
        else
        {
          m_goldDir = optsMap["goldDir"].as<std::string>();
        }
        if ( !dp::util::directoryExists( m_goldDir ) )
        {
          if ( ! dp::util::createDirectory( m_goldDir ) )
          {
            std::cerr << "Unable to create directory <" << m_goldDir << ">\n";
            return( false );
          }
        }

        // Image Dir
        if( optsMap["imageDir"].empty() )
        {
          std::cerr << "Warning: No image directory was specified\n";
        }
        else
        {
          m_imageDir = optsMap["imageDir"].as<std::string>();
        }
        if ( !dp::util::directoryExists( m_imageDir ) )
        {
          if ( ! dp::util::createDirectory( m_imageDir ) )
          {
            std::cerr << "Unable to create directory <" << m_imageDir << ">\n";
            return( false );
          }
        }

        // Filename Prefix

        if( !optsMap["outputFilenamePrefix"].empty() )
        {
          m_filenamePrefix = optsMap["outputFilenamePrefix"].as<std::string>();
        }

        // Filename Suffix

        if( !optsMap["outputFilenameSuffix"].empty() )
        {
          m_filenameSuffix = optsMap["outputFilenameSuffix"].as<std::string>();
        }

        return true;
      }

      bool MeasurementFunctorGoldImage::callOnRun( unsigned int i )
      {
        bool ok = MeasurementFunctor::callOnRun( i );
        takeScreenshot();
        return( ok );
      }

      bool MeasurementFunctorGoldImage::callOnClear()
      {
        bool goldFunctionSucceeded = saveImages();
        bool testSucceeded = MeasurementFunctor::callOnClear();
        return( goldFunctionSucceeded && testSucceeded );
      }

      bool MeasurementFunctorGoldImage::saveImages()
      {
        bool foundFile = true;
        bool success = true;

        const std::string& name = getCurTestName();

        unsigned int nShots = dp::util::checked_cast<unsigned int>(m_screenshots.size());
        for(unsigned int i = 0; i < nShots; i++)
        {
          std::string goldImageName(m_filenamePrefix + name + "_" + getTest()->getDescriptionOnRunInit(i) + m_filenameSuffix + util::to_string(".png"));

          if( m_gold )
          {

            std::string filename( m_goldDir + "\\" + goldImageName );
            dp::util::imageToFile( retrieveScreenshot(i), filename );
            std::cerr << "Saved gold image: " << goldImageName + "\n";
          }
          else
          { 
            util::ImageSharedPtr gold = util::imageFromFile(m_goldDir + "\\" + goldImageName);
            if( !gold )
            {
              if(foundFile)
              {
                std::cerr << "Could not find the following gold images\n";
                foundFile = false;
              }
              std::cerr << goldImageName << "\n";
              continue;
            }


            if( *retrieveScreenshot(i) != *gold )
            {
              std::cerr << "Gold image comparison failed for " << goldImageName << "\n";
              success = false;
            }

            std::string filename( m_imageDir + "\\" + goldImageName );
            dp::util::imageToFile(retrieveScreenshot(i), filename);
          }
        }

        m_screenshots.clear();

        return foundFile && success;
      }

      void MeasurementFunctorGoldImage::takeScreenshot()
      {
        DP_ASSERT( dynamic_cast<TestRender*>(getTest()) );
        m_screenshots.push_back( static_cast<TestRender*>(getTest())->getScreenshot() );
      }

      util::ImageSharedPtr MeasurementFunctorGoldImage::retrieveScreenshot( unsigned int i )
      {
        return m_screenshots[i];
      }

      MeasurementFunctorGoldImage::~MeasurementFunctorGoldImage()
      {

      }

    } // namespace core
  } // namespace testfw
} // namespace dp
