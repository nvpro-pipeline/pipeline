// Copyright NVIDIA Corporation 2002-2005
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


// WARNING!!! Stay clear of binary_little_endian files which contain "comment VCGLIB generated"!!!
// I have found multiple PLY files on AIM SHAPE which were containing broken little endian data blocks.
// They contained carriage return line feed pairs (0x0D 0x0A) in the middle of the binary data, 
// which most likely happened because the files have either been written under Windows in "w" (write ascii), 
// not "wb" (write binary) access mode or been transfered in ASCII mode via FTP. Trying to reconvert those back, didn't work either.
// In either case binary files are corrupted afterwards. ASCII files survive that.


#include  <cstdio> // make mingw happy
#include  <dp/Exception.h>
#include  <dp/sg/core/Config.h>
#include  <dp/sg/core/GeoNode.h>
#include  <dp/sg/core/IndexSet.h>
#include  <dp/sg/core/Scene.h>
#include  <dp/sg/core/VertexAttributeSet.h>
#include  <dp/sg/algorithm/SmoothTraverser.h>
#include  <dp/sg/io/PlugInterfaceID.h>
#include  <dp/util/File.h>

#include  "PLYLoader.h"

using namespace dp::math;
using namespace dp::util;
using namespace dp::sg::core;

using std::map;
using std::pair;
using std::string;
using std::vector;

const UPITID PITID_SCENE_LOADER(UPITID_SCENE_LOADER, UPITID_VERSION); // plug-in type
UPIID PIID_PLY_SCENE_LOADER = UPIID(".PLY", PITID_SCENE_LOADER); 

#if defined( _WIN32 )
// is this necessary??
BOOL APIENTRY DllMain(HANDLE hModule, DWORD reason, LPVOID lpReserved)
{
  if (reason == DLL_PROCESS_ATTACH)
  {
    // initialize supported Plug Interface ID
    PIID_PLY_SCENE_LOADER = UPIID(".PLY", PITID_SCENE_LOADER); 
    int i=0;
  }

  return TRUE;
}
#elif defined( LINUX )
void lib_init()
{
  int i=0;
}
#endif

bool getPlugInterface(const UPIID& piid, dp::util::PlugInSharedPtr & pi)
{
  // Check if UPIID is properly initialized. 
  // DP_ASSERT(PIID_PLY_SCENE_LOADER==UPIID(".PLY", PITID_SCENE_LOADER));
  
  if ( piid==PIID_PLY_SCENE_LOADER )
  {
    pi = PLYLoader::create();
    return( !!pi );
  }
  return false;
}

void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids )
{
  piids.clear();

  piids.push_back(PIID_PLY_SCENE_LOADER);
}

PLYElement::PLYElement()
: name("")
, count(0)
{
}

PLYElement::~PLYElement()
{
  std::vector<PLYProperty *>::iterator itpProp;

  for (itpProp = m_pProperties.begin(); itpProp != m_pProperties.end(); itpProp++)
  {
    delete *itpProp;
  }
  m_pProperties.clear();
}



PLYProperty::PLYProperty()
: name("")
, countType(PLY_TOKEN_UNKNOWN)
, dataType(PLY_TOKEN_UNKNOWN)
, pfnReadCount(NULL)
, pfnReadData(NULL)
, pfnReadAttribute(NULL)
, index(PLY_USER_DEFINED_COMPONENT)
{
}

PLYProperty::~PLYProperty()
{
}


PLYLoaderSharedPtr PLYLoader::create()
{
  return( std::shared_ptr<PLYLoader>( new PLYLoader() ) );
}

PLYLoader::PLYLoader()
: m_fm(NULL)
, m_pcCurrent(NULL)
, m_pcEOF(NULL)
, m_plyFormat(0)
, m_line(0) // Not really used.
{
  m_token[0] = '\0'; // Empty string.

  // Table of attribute data reading functions.
  // Special in that they convert integer data to float according to the OpenGL specs.
  m_apfnReadAttribute[0][0] = &PLYLoader::readAttributeAscii_CHAR;  
  m_apfnReadAttribute[0][1] = &PLYLoader::readAttributeAnyEndian_CHAR; // !!!
  m_apfnReadAttribute[0][2] = &PLYLoader::readAttributeAnyEndian_CHAR; // !!!
      
  m_apfnReadAttribute[1][0] = &PLYLoader::readAttributeAscii_UCHAR;
  m_apfnReadAttribute[1][1] = &PLYLoader::readAttributeAnyEndian_UCHAR; // !!!
  m_apfnReadAttribute[1][2] = &PLYLoader::readAttributeAnyEndian_UCHAR; // !!!

  m_apfnReadAttribute[2][0] = &PLYLoader::readAttributeAscii_SHORT;
  m_apfnReadAttribute[2][1] = &PLYLoader::readAttributeLittleEndian_SHORT;
  m_apfnReadAttribute[2][2] = &PLYLoader::readAttributeBigEndian_SHORT;

  m_apfnReadAttribute[3][0] = &PLYLoader::readAttributeAscii_USHORT;
  m_apfnReadAttribute[3][1] = &PLYLoader::readAttributeLittleEndian_USHORT;
  m_apfnReadAttribute[3][2] = &PLYLoader::readAttributeBigEndian_USHORT;

  m_apfnReadAttribute[4][0] = &PLYLoader::readAttributeAscii_INT;
  m_apfnReadAttribute[4][1] = &PLYLoader::readAttributeLittleEndian_INT;
  m_apfnReadAttribute[4][2] = &PLYLoader::readAttributeBigEndian_INT;

  m_apfnReadAttribute[5][0] = &PLYLoader::readAttributeAscii_UINT;
  m_apfnReadAttribute[5][1] = &PLYLoader::readAttributeLittleEndian_UINT;
  m_apfnReadAttribute[5][2] = &PLYLoader::readAttributeBigEndian_UINT;

  m_apfnReadAttribute[6][0] = &PLYLoader::readAttributeAscii_FLOAT;
  m_apfnReadAttribute[6][1] = &PLYLoader::readAttributeLittleEndian_FLOAT;
  m_apfnReadAttribute[6][2] = &PLYLoader::readAttributeBigEndian_FLOAT;

  m_apfnReadAttribute[7][0] = &PLYLoader::readAttributeAscii_FLOAT; // !!!
  m_apfnReadAttribute[7][1] = &PLYLoader::readAttributeLittleEndian_DOUBLE;
  m_apfnReadAttribute[7][2] = &PLYLoader::readAttributeBigEndian_DOUBLE;


  // Read data as it is. Caller needs to figure out what it was.
  m_apfnRead[0][0] = &PLYLoader::readAscii_CHAR;  
  m_apfnRead[0][1] = &PLYLoader::readAnyEndian_CHAR; // !!!
  m_apfnRead[0][2] = &PLYLoader::readAnyEndian_CHAR; // !!!
      
  m_apfnRead[1][0] = &PLYLoader::readAscii_UCHAR;
  m_apfnRead[1][1] = &PLYLoader::readAnyEndian_UCHAR; // !!!
  m_apfnRead[1][2] = &PLYLoader::readAnyEndian_UCHAR; // !!!

  m_apfnRead[2][0] = &PLYLoader::readAscii_SHORT;
  m_apfnRead[2][1] = &PLYLoader::readLittleEndian_SHORT;
  m_apfnRead[2][2] = &PLYLoader::readBigEndian_SHORT;

  m_apfnRead[3][0] = &PLYLoader::readAscii_USHORT;
  m_apfnRead[3][1] = &PLYLoader::readLittleEndian_USHORT;
  m_apfnRead[3][2] = &PLYLoader::readBigEndian_USHORT;

  m_apfnRead[4][0] = &PLYLoader::readAscii_INT;
  m_apfnRead[4][1] = &PLYLoader::readLittleEndian_INT;
  m_apfnRead[4][2] = &PLYLoader::readBigEndian_INT;

  m_apfnRead[5][0] = &PLYLoader::readAscii_UINT;
  m_apfnRead[5][1] = &PLYLoader::readLittleEndian_UINT;
  m_apfnRead[5][2] = &PLYLoader::readBigEndian_UINT;

  m_apfnRead[6][0] = &PLYLoader::readAscii_FLOAT;
  m_apfnRead[6][1] = &PLYLoader::readLittleEndian_FLOAT;
  m_apfnRead[6][2] = &PLYLoader::readBigEndian_FLOAT;

  m_apfnRead[7][0] = &PLYLoader::readAscii_DOUBLE;
  m_apfnRead[7][1] = &PLYLoader::readLittleEndian_DOUBLE;
  m_apfnRead[7][2] = &PLYLoader::readBigEndian_DOUBLE;
}

PLYLoader::~PLYLoader()
{
  cleanup();
}

void PLYLoader::cleanup( void )
{
  delete m_fm;
  m_fm = NULL;
  m_mapStringToToken.clear();
  m_mapStringToAttributeComponent.clear();
  m_pcCurrent = NULL;
  m_pcEOF = NULL;
  m_token[0] = '\0';
  m_plyFormat = 0;  
  std::vector<PLYElement *>::iterator itpEle;
  for (itpEle = m_pElements.begin(); itpEle != m_pElements.end(); itpEle++)
  {
    delete *itpEle;
  }
  m_pElements.clear();
  m_line = 0;
  m_searchPaths.clear();
}

// Finds the next ASCII token, copies its string into m_token and
// returns the count of bytes processed inside m_pcCurrent for this.
// m_pcCurrent is not changed.
// Returns zero if there is an unexpected EOF condition.
int PLYLoader::lookAheadToken(void)
{
  const char whitespace[] = " \r\n\t\0";
  char *pcWork = m_pcCurrent;
  char *pcTokenStart;

  // Skip whitespace
  while ((pcWork < m_pcEOF) && 
         strchr(whitespace, *pcWork))
  {
    pcWork++;
  }
  onUnexpectedEndOfFile(pcWork >= m_pcEOF);

  // Remember token start address
  pcTokenStart = pcWork;
  while ((pcWork < m_pcEOF) && 
         !strchr(whitespace, *pcWork))
  {
    pcWork++;
  }
  onUnexpectedEndOfFile(pcWork >= m_pcEOF);

  // Copy the found token into a c-string (not editing the file mapping).
  int size = static_cast<int>(pcWork - pcTokenStart);
  DP_ASSERT(size + 1 < 256);

  strncpy(m_token, pcTokenStart, size);
  m_token[size] = '\0';

  return static_cast<int>(pcWork - m_pcCurrent); // advance includes the skipped whitespace!
}


int PLYLoader::skipLine(void)
{
  char *pcWork = m_pcCurrent;

  // Skip everything until carriage return or linefeed is found.
  // The spec names carriage return as end of line character, 
  // but Unix uses linefeed and Window carriage return plus linefeed.
  // Just skip anything until either is found.
  // The next getNextToken() will ignore possibly remaining line end 
  // characters as whitespace.
  while ((pcWork < m_pcEOF) && 
         (*pcWork != '\r') && 
         (*pcWork != '\n'))
  {
    pcWork++;
  }
  onUnexpectedEndOfFile(pcWork >= m_pcEOF);

  return (int)(pcWork - m_pcCurrent);
}


/* 
  A PLY header using the old data type names could look like this:

  ply
  format ascii 1.0
  comment This line is ignored.
  element vertex 24
  property float x
  property float y
  property float z
  property float nx
  property float ny
  property float nz
  property uchar red
  property uchar green
  property uchar blue
  element face 36
  property list uchar int vertex_indices
  end_header
*/

// SceneLoader API
SceneSharedPtr PLYLoader::load(const string& filename, const vector<string> &searchPaths, dp::sg::ui::ViewStateSharedPtr & viewState)
{
  if ( !dp::util::fileExists(filename) )
  {
    throw dp::FileNotFoundException( filename );
  }

  bool success = true;

  cleanup(); 

  PLYElement  *curElement  = NULL;
  PLYProperty *curProperty = NULL;
  std::vector<PLYElement *>::iterator itpEle;
  std::vector<PLYProperty *>::iterator itpProp;

  SceneSharedPtr sceneResult;
  
  // Make a copy of the given search paths.
  m_searchPaths = searchPaths;

  string localPath = dp::util::getFilePath( filename );
  m_searchPaths.insert(m_searchPaths.begin(), localPath);

  try
  {
    // the resulting file name should be valid if we get here
    DP_ASSERT(!filename.empty());

    size_t filesize = dp::util::fileSize( filename );
    if ( filesize != -1 )
    {
      // Stuff needed to fill the scene in the end.
      int hasAttribute = 0;

      // Now the elements and properties are fully setup.
      // Read the vertex attributes into local vectors:
      std::vector<Vec3f> vertex;
      std::vector<Vec3f> normal;
      std::vector<Vec3f> color;
      // and read the faces and split them to an individual triangle list:
      std::vector<unsigned int> indices;

      unsigned int numVertices = 0;
      unsigned int numFaces    = 0;

      // Map the file into our address space.
      m_fm = new ReadMapping(filename);

      if (m_fm->isValid())
      {
        // Get access to the whole file in binary form as an unsigned char pointer.
        Offset_AutoPtr<char> pcFileMapping(m_fm, callback(), 0, filesize);

        // Parse the PLY header which defines the vertex attribute layout and data format in the file.
        // Construct the necessary read function list to read the data into local data structures
        // which will then be put into scene nodes.
        if (pcFileMapping)
        {
          // Prepare everything for header parsing.
          initializeMapStringToToken();

          // Set variables which determin the parsing and end of file.
          m_pcCurrent = pcFileMapping.operator->();
          m_pcEOF     = m_pcCurrent + filesize;

          PLY_PARSER_STATE state = PLY_STATE_PLY;

          // Parsing the header
          while (success && m_pcCurrent < m_pcEOF && state != PLY_STATE_END)
          {
            int advance = lookAheadToken();
            if (advance)
            {
              PLY_TOKEN idToken;
              std::string token = m_token;
                
              // Look up the parsed token and convert it into an enum for the state machine.
              std::map<std::string, PLY_TOKEN>::const_iterator it = m_mapStringToToken.find(token);
              if (it != m_mapStringToToken.end())
              {
                idToken = it->second;
              }
              else
              {
                idToken = PLY_TOKEN_UNKNOWN; // Element and property identifiers will map to this.
              }

              if (idToken == PLY_TOKEN_COMMENT || 
                  idToken == PLY_TOKEN_OBJINFO) // Not supporting obj_info, handle it as comment.
              {
                m_pcCurrent += advance; // Skip what we tokenized in that line so far. 
                                        // Otherwise we would see the last linefeed again and skipLine would do nothing.
                advance = skipLine();   // Skip the whole line until \r or \n. (Could add previously found advance here.)
              }
              else
              {
                // Small state machine which handles the header parsing.
                switch (state)
                {
                  case PLY_STATE_PLY: // PLY file header identifier.
                    if (idToken == PLY_TOKEN_PLY)
                    {
                      state = PLY_STATE_FORMAT;
                    }
                    else
                    {
                      success = false;
                      onUnexpectedToken("ply", token);
                    }
                    break;

                  case PLY_STATE_FORMAT:
                    if (idToken == PLY_TOKEN_FORMAT)
                    {
                      state = PLY_STATE_FORMAT_TYPE;
                    }
                    else
                    {
                      success = false;
                      onUnexpectedToken("format", token);
                    }
                    break;

                  case PLY_STATE_FORMAT_TYPE:
                    if (idToken == PLY_TOKEN_ASCII || 
                        idToken == PLY_TOKEN_BINARYLITTLEENDIAN ||
                        idToken == PLY_TOKEN_BINARYBIGENDIAN)
                    {
                      // Convert to [0,2] index (ascii/little/big) for table lookups.
                      m_plyFormat = idToken - PLY_TOKEN_ASCII; 
                      state = PLY_STATE_FORMAT_VERSION;
                    }
                    else
                    {
                      success = false;
                      onUnexpectedToken("ascii|binary_big_endian|binary_little_endian", token);
                    }
                    break;

                  case PLY_STATE_FORMAT_VERSION:
                    if (idToken == PLY_TOKEN_ONEPOINTZERO)
                    {
                      state = PLY_STATE_ELEMENT;
                    }
                    else
                    {
                      state = PLY_STATE_ELEMENT; // Try to continue.
                      onUnsupportedToken("expected format version 1.0, ignored", token);
                    }
                    break;

                  case PLY_STATE_ELEMENT:
                    if (idToken == PLY_TOKEN_ELEMENT)
                    {
                      // Generate a new element which will be filled with data 
                      // until the next element or end_header is found.
                      curElement = new PLYElement;
                      if (!curElement)
                      {
                        success = false;
                        onError("Out of memory while allocating PLYElement");
                      }
                      state = PLY_STATE_ELEMENT_IDENTIFIER;
                    }
                    else
                    {
                      success = false;
                      onUnexpectedToken("element", token);
                    }
                    break;

                  case PLY_STATE_ELEMENT_IDENTIFIER:
                    // It's just a name, figure out later what data we can handle.
                    curElement->name = m_token;
                    state = PLY_STATE_ELEMENT_COUNT;
                    break;

                  case PLY_STATE_ELEMENT_COUNT:
                    curElement->count = atoi(m_token);
                    state = PLY_STATE_PROPERTY;
                    break;

                  case PLY_STATE_PROPERTY:
                    // Whatever we find in this state, if we have seen a property before, 
                    // that one must be completely specified here (or something is wrong
                    // with the header, which we'll find out later) and attached to the 
                    // current element now.
                    if (curProperty)
                    {
                      curElement->m_pProperties.push_back(curProperty);
                      curProperty = NULL;
                    }

                    if (idToken == PLY_TOKEN_PROPERTY)
                    {
                      curProperty = new PLYProperty;
                      if (!curProperty)
                      {
                        success = false;
                        onError("Out of memory while allocating PLYProperty");
                      }
                      state = PLY_STATE_PROPERTY_TYPE; // data type or "list".
                    }
                    else // If there is not a property anymore, store the current element.
                    {
                      // Not continuing with a further property in this state means 
                      // we we have fully specified an element. 
                      // Add it to the list of elements.
                      m_pElements.push_back(curElement);
                      curElement = NULL;
                       
                      // Determine where the state machine needs to continue.
                      // Only element or end_header should follow here.
                      // Comments or obj_type don't reach this.
                      if (idToken == PLY_TOKEN_ELEMENT)
                      {
                        state = PLY_STATE_ELEMENT;  // Start a new element description.
                        advance = 0;  // Do not consume this element token, read it again at a different state. 
                                      // That's why the function is named "look ahead".
                      }
                      else if (idToken == PLY_TOKEN_ENDHEADER)
                      {
                        state = PLY_STATE_END; // Exit the header parsing.
                      }
                      else // Remember, comments and obj_type tokens don't reach this!
                      {
                        success = false;
                        onUnexpectedToken("property|element|end_header", token);
                      }
                    }
                    break;

                  case PLY_STATE_PROPERTY_TYPE: // scalar or list
                    switch (idToken)
                    {
                      // Data type directly behind property keyword means scalar data.
                      case PLY_TOKEN_CHAR:
                      case PLY_TOKEN_UCHAR:
                      case PLY_TOKEN_SHORT:
                      case PLY_TOKEN_USHORT:
                      case PLY_TOKEN_INT:
                      case PLY_TOKEN_UINT:
                      case PLY_TOKEN_FLOAT:
                      case PLY_TOKEN_DOUBLE:
                        curProperty->countType = PLY_TOKEN_UNKNOWN; // Not a list.
                        curProperty->dataType  = idToken;
                        // Could be user defined unknown data, that is skipped by the read as-is function in pfnReadData.
                        curProperty->pfnReadData      = m_apfnRead[idToken][m_plyFormat];
                        curProperty->pfnReadAttribute = m_apfnReadAttribute[idToken][m_plyFormat];
                        state = PLY_STATE_PROPERTY_NAME;
                        break;

                      case PLY_TOKEN_LIST:
                        state = PLY_STATE_PROPERTY_LIST_COUNT_TYPE;
                        break;

                      default:
                        success = false;
                        onUnsupportedToken("expected data type or list keyword", m_token);
                        break;
                    }
                    break;

                  case PLY_STATE_PROPERTY_LIST_COUNT_TYPE: // integer scalar counts only.
                    switch (idToken)
                    {
                      case PLY_TOKEN_CHAR:
                      case PLY_TOKEN_UCHAR:
                      case PLY_TOKEN_SHORT:
                      case PLY_TOKEN_USHORT:
                      case PLY_TOKEN_INT:
                      case PLY_TOKEN_UINT:
                        curProperty->countType = idToken;
                        curProperty->pfnReadCount = m_apfnRead[idToken][m_plyFormat];
                        state = PLY_STATE_PROPERTY_LIST_DATA_TYPE;
                        break;

                      default:
                        success = false;
                        onUnsupportedToken("expected integer data type for list count type", m_token);
                        break;
                    }
                    break;

                  case PLY_STATE_PROPERTY_LIST_DATA_TYPE: // scalar data type
                    switch (idToken)
                    {
                      case PLY_TOKEN_CHAR:
                      case PLY_TOKEN_UCHAR:
                      case PLY_TOKEN_SHORT:
                      case PLY_TOKEN_USHORT:
                      case PLY_TOKEN_INT:
                      case PLY_TOKEN_UINT:
                      case PLY_TOKEN_FLOAT:
                      case PLY_TOKEN_DOUBLE:
                        curProperty->dataType = idToken;
                        curProperty->pfnReadData = m_apfnRead[idToken][m_plyFormat];
                        state = PLY_STATE_PROPERTY_NAME;
                        break;

                      default:
                        success = false;
                        onUnsupportedToken("expected scalar data type for list data type", m_token);
                        break;
                    }
                    break;

                  case PLY_STATE_PROPERTY_NAME: // Determines where the data goes.
                    curProperty->name = m_token;

                    // Determine the destination data offset inside of a vertex attribute component.
                    // So far I've only seen vertex attribute properties labeled with x, y, z, nx, ny, nz, 
                    // so support those, read the rest to dev/nul.
                    // Reading vertex attribute data is done to a local vertex attribute consisting of a six float array for now.
                    if (curElement->name == "vertex")
                    {
                      PLY_ATTRIBUTE_COMPONENT component = PLY_USER_DEFINED_COMPONENT; // Assume we don't know what it is.

                      std::map<std::string, PLY_ATTRIBUTE_COMPONENT>::const_iterator it = 
                        m_mapStringToAttributeComponent.find(curProperty->name);

                      if (it != m_mapStringToAttributeComponent.end())
                      {
                        component = it->second;
                      }
                        
                      // Enums are zero based and indices and bit position in the hasAttribute flag match accordingly.
                      // where "user defined" components all go into the ignored slot.
                      if (PLY_VERTEX_X <= component && component <= PLY_USER_DEFINED_COMPONENT) 
                      {
                        curProperty->index = component;
                        hasAttribute |= (1 << component);
                      }
                      else
                      {
                        // Internal error! Must not happen or some code wasn't correctly updated (enum or map).  
                        DP_ASSERT(0);
                        // Reading that data would access the attribute array out of bounds, 
                        // so force it into the user defined slot.
                        curProperty->index = PLY_USER_DEFINED_COMPONENT;
                        hasAttribute |= (1 << PLY_USER_DEFINED_COMPONENT);
                        success = false;
                      }
                    }
                    state = PLY_STATE_PROPERTY; // Expect more properties.
                    break;

                  default:
                    success = false;
                    onUnexpectedToken("", token);
                    break;
                }
              }
              m_pcCurrent += advance;
            }
            else // Haven't found a token.
            {
              success = false;
              onEmptyToken("No token found", "");
            }
          }
            
          // If this is still set here, we have not reached the end state correctly 
          // and the allocated data has not been transferred to the current element.
          if (curProperty)
          {
            delete curProperty;
            curProperty = NULL;
          }

          // Similarly for the current element, if that is found != NULL here,
          // it has not been transferred to the m_pElements vector.
          if (curElement)
          {
            // First delete all attached properties.
            for (itpProp = curElement->m_pProperties.begin(); itpProp != curElement->m_pProperties.end(); itpProp++)
            {
              delete *itpProp;
            }
            curElement->m_pProperties.clear();
            // Then the container.
            delete curElement;
            curElement = NULL;
          }
           

          if (success && state == PLY_STATE_END)
          {
            // The question is now if the file contains \r or \n or \r\n as nextline.
            // The PLY "spec" says \r, Linux could have \n, but the problematic case is \r\n under Windows in a binary file.
            // (ASCII is not a problem.)
            // When there is binary data containing \n as byte we would inadvertently use the wrong reading start position.
            if (m_pcCurrent < m_pcEOF - 2) // At least two chars left.
            {
              if (m_plyFormat) // Only binary files need to skip the line delimiter. ASCII just eats the whitespace away.
              {
                if (m_pcCurrent[0] == '\r')
                {
                  m_pcCurrent++; // Skip the expected carriage return \r.
                }
                if (m_pcCurrent[0] == '\n') // Unix file with linefeed \n or Windows file with \r\n?
                {
                  m_pcCurrent++; // If there happens to be carriage return \r only and 
                                  // a binary data byte 10 following, we're screwed here!!!
                  // On the other hand I have seen tons of corrupted PLY files 
                  // which had CRLF in the middle of the binary data blocks. 
                  // All those will not work or even crash.
                }
              }
            }
            else
            {
              DP_ASSERT(0); // Broken file. No data behind header.
              success = false;
              onUnexpectedEndOfFile(true);
            }
          }
          else
          {
            DP_ASSERT(0); // Header reading failed.
            success = false;
            onUnexpectedEndOfFile(true);
          }

          if (success)
          {
            // Make sure we have vertex data. 
            // We convert anything to Vertex3f, so it just matters that we have at least one vertex component to continue.
            DP_ASSERT((hasAttribute & ATTRIBUTE_MASK_VERTEX));
            // Make sure we either have no or fully specified normals.
            DP_ASSERT((hasAttribute & ATTRIBUTE_MASK_NORMAL) == 0 || 
                        (hasAttribute & ATTRIBUTE_MASK_NORMAL) == ATTRIBUTE_MASK_NORMAL);
            // Make sure we either have no or fully specified colors.
            DP_ASSERT((hasAttribute & ATTRIBUTE_MASK_COLOR) == 0 || 
                        (hasAttribute & ATTRIBUTE_MASK_COLOR) == ATTRIBUTE_MASK_COLOR);

            float attributes[PLY_USER_DEFINED_COMPONENT + 1] = {0.0f}; // All unused components are 0.0f.


            for (itpEle = m_pElements.begin(); itpEle != m_pElements.end(); itpEle++)
            {
              if ((*itpEle)->name == "vertex") // Vertex attribute data description inside the properties.
              {
                DP_ASSERT(numVertices == 0); // Make sure the file has only one vertex element.
                numVertices = (*itpEle)->count;

                // Resize attribute vectors to expected amount.
                if (hasAttribute & ATTRIBUTE_MASK_VERTEX) // Vertex3f
                {
                  vertex.resize(numVertices);
                }
                if (hasAttribute & ATTRIBUTE_MASK_NORMAL) // Normal3f
                {
                  normal.resize(numVertices);
                }
                if (hasAttribute & ATTRIBUTE_MASK_COLOR)  // Color3f
                {
                  color.resize(numVertices);
                }

                for (unsigned int i = 0; i < numVertices; i++)
                {
                  for (itpProp = (*itpEle)->m_pProperties.begin(); itpProp != (*itpEle)->m_pProperties.end(); itpProp++)
                  {
                    (this->*((*itpProp)->pfnReadAttribute))(&attributes[(*itpProp)->index]);
                  }

                  if (hasAttribute & ATTRIBUTE_MASK_VERTEX) // Vertex3f
                  {
                    setVec(vertex[i], attributes[PLY_VERTEX_X], attributes[PLY_VERTEX_Y], attributes[PLY_VERTEX_Z]);
                  }
                  if (hasAttribute & ATTRIBUTE_MASK_NORMAL) // Normal3f
                  {
                    setVec(normal[i], attributes[PLY_NORMAL_X], attributes[PLY_NORMAL_Y], attributes[PLY_NORMAL_Z]);
                  }
                  if (hasAttribute & ATTRIBUTE_MASK_COLOR) // Color3f
                  {
                    setVec(color[i], attributes[PLY_COLOR_R], attributes[PLY_COLOR_G], attributes[PLY_COLOR_B]);
                  }
                }
              }
              else if ((*itpEle)->name == "face") // Face data description inside the properties.
              {
                DP_ASSERT(numFaces == 0); // Make sure the file has only one face element.
                numFaces = (*itpEle)->count;

                indices.reserve( 3 * numFaces );  // This assumes triangles. Will dynamically increase (slowdown) if there is a lot of tesselation happening.

                for (unsigned int i = 0; i < numFaces; i++)
                {
                  for (itpProp = (*itpEle)->m_pProperties.begin(); itpProp != (*itpEle)->m_pProperties.end(); itpProp++)
                  {
                    if ((*itpProp)->name == "vertex_indices") 
                    {
                      Vec3ui face;
                      unsigned int count = readListCounterOrIndex((*itpProp)->pfnReadCount, (*itpProp)->countType);

                      if (count >= 3) 
                      {
                        face[0] = readListCounterOrIndex((*itpProp)->pfnReadData, (*itpProp)->dataType);
                        face[1] = readListCounterOrIndex((*itpProp)->pfnReadData, (*itpProp)->dataType);
                        face[2] = readListCounterOrIndex((*itpProp)->pfnReadData, (*itpProp)->dataType);
                          
                        if (face[0] >= numVertices)
                        {
                          success = false;
                          onInvalidValue((int) face[0], "Vertex index outside vertex pool size.\n(Binary file ASCII transferred?)", "vertex_indices");
                        }
                        if (face[1] >= numVertices)
                        {
                          success = false;
                          onInvalidValue((int) face[1], "Vertex index outside vertex pool size.\n(Binary file ASCII transferred?)", "vertex_indices");
                        }
                        if (face[2] >= numVertices)
                        {
                          success = false;
                          onInvalidValue((int) face[2], "Vertex index outside vertex pool size.\n(Binary file ASCII transferred?)", "vertex_indices");
                        }
                        if (success)
                        {
                          indices.push_back( face[0] );
                          indices.push_back( face[1] );
                          indices.push_back( face[2] );
                        }
                          
                        // Decompose into triangles, (triangle_fan like).
                        for (unsigned int j = 3; j < count; j++)
                        {
                          face[1] = face[2];
                          face[2] = readListCounterOrIndex((*itpProp)->pfnReadData, (*itpProp)->dataType);
                          if (face[2] >= numVertices)
                          {
                            success = false;
                            onInvalidValue((int) face[2], "Vertex index outside vertex pool size.\n(Binary file ASCII transferred?)", "vertex_indices");
                          }
                          if (success)
                          {
                            indices.push_back( face[0] );
                            indices.push_back( face[1] );
                            indices.push_back( face[2] );
                          }
                        }
                      }
                      else
                      {
                        // DP_ASSERT(0); // Should not happen.
                        // Remove degenerated faces.
                        for (unsigned int j = 0; j < count; j++)
                        {
                          (void) readListCounterOrIndex((*itpProp)->pfnReadData, (*itpProp)->dataType);
                        }
                      }
                    }
                    else // Skip unknown face properties. // For example the armadillo file has an uchar intensity per face.
                    {
                      ignoreProperty(*itpProp);
                    }
                  } // properties
                } // numFaces
              }
              else // Unsupported element, skip-read it.
              {
                // DP_ASSERT(0); // Find files which do this.
                unsigned int numElements = (*itpEle)->count;

                for (unsigned int i = 0; i < numElements; i++)
                {
                  for (itpProp = (*itpEle)->m_pProperties.begin(); itpProp != (*itpEle)->m_pProperties.end(); itpProp++)
                  {
                    ignoreProperty(*itpProp);
                  }
                }
              }
            }
          }
          else // file mapping failed
          {
            success = false;
            onUnexpectedEndOfFile(true);
          }
        } // success
      }

      delete m_fm; // Done with the filemapping.
      m_fm = NULL;

      if (success)
      {
        IndexSetSharedPtr iset( IndexSet::create() );
        iset->setData( &indices[0], checked_cast<unsigned int>(indices.size()) );

        bool generateNormals = false;
        VertexAttributeSetSharedPtr cvas = VertexAttributeSet::create();
        DP_ASSERT(vertex.size() == numVertices && numVertices > 0);
        if (vertex.size())
        {
          cvas->setVertices(&vertex[0], static_cast<unsigned int>(vertex.size()));
        }
        if (normal.size())
        {
          cvas->setNormals(&normal[0], static_cast<unsigned int>(normal.size()));
        }
        else
        {
          // Do a cheap face normal generation if the there are no normals in the model.
          // That's avoids assertions or crashes in cases the material editor assigns 
          // CgFX effects calling generateTangntSpace expect normals.
          generateNormals = true;
        }
        if (color.size())
        {
          cvas->setColors(&color[0], static_cast<unsigned int>(color.size()));
        }

        // Generate the scene from the gathered data.
        PrimitiveSharedPtr pTriangles = Primitive::create( PRIMITIVE_TRIANGLES );
        pTriangles->setIndexSet( iset );
        pTriangles->setVertexAttributeSet( cvas );
        if ( generateNormals )
        {
          pTriangles->generateNormals();
        }

        // Face and attribute data has been copied into the scenegraph.
        // Clear the local arrays to save memory for the SmoothTraverser.
        indices.clear();
        vertex.clear();
        normal.clear();
        color.clear();
      
        GeoNodeSharedPtr pGeoNode = GeoNode::create();
        pGeoNode->setPrimitive( pTriangles );

        // Create top-level scene.
        sceneResult = Scene::create();
          
        // Add group as scene's top-level.
        sceneResult->setRootNode(pGeoNode);
      } // sucess
    }
    else // Couldn't determine filesize.
    {
      success = false;
      onUnexpectedEndOfFile(true);
    }
  } // try

  catch (...) // note: stack unwinding doesn't consider heap objects!
  {
    // If this is still set here, we have not reached the end state correctly 
    // and the allocated data has not been transferred to the current element.
    if (curProperty)
    {
      delete curProperty;
      curProperty = NULL;
    }

    // Similarly for the current element, if that is found != NULL here,
    // it has not been transferred to the m_pElements vector.
    if (curElement)
    {
      // First delete all attached properties.
      for (itpProp = curElement->m_pProperties.begin(); itpProp != curElement->m_pProperties.end(); itpProp++)
      {
        delete *itpProp;
      }
      curElement->m_pProperties.clear();
      // Then the container.
      delete curElement;
      curElement = NULL;
    }

    cleanup();

    throw;
  }

  cleanup();

  return sceneResult;
}


void PLYLoader::ignoreProperty(PLYProperty *p)
{
  union
  {
    char c;
    unsigned char uc;
    short s;
    unsigned short us;
    int i;
    unsigned int ui;
    float f;
    double d;
  } ignore;

  if (p->pfnReadCount) // A list of something.
  {
    unsigned int count = readListCounterOrIndex(p->pfnReadCount, p->countType);
    for (unsigned int j = 0; j < count; j++)
    {
      (this->*(p->pfnReadData))(&ignore);
    }
  }
  else if (p->pfnReadData) // Scalar data of something.
  {
    (this->*(p->pfnReadData))(&ignore);
  }
}

void PLYLoader::initializeMapStringToToken(void)
{
  m_mapStringToToken.clear();
  
  // "Magic" file identifier:
  m_mapStringToToken.insert(std::make_pair("ply", PLY_TOKEN_PLY)); 
  // Format description of data folowing the header:
  m_mapStringToToken.insert(std::make_pair("format", PLY_TOKEN_FORMAT));
  m_mapStringToToken.insert(std::make_pair("ascii", PLY_TOKEN_ASCII));
  m_mapStringToToken.insert(std::make_pair("binary_little_endian", PLY_TOKEN_BINARYLITTLEENDIAN));
  m_mapStringToToken.insert(std::make_pair("binary_big_endian", PLY_TOKEN_BINARYBIGENDIAN));
  m_mapStringToToken.insert(std::make_pair("1.0", PLY_TOKEN_ONEPOINTZERO));
  // Comments and annotations, skipped in this loader:
  m_mapStringToToken.insert(std::make_pair("comment", PLY_TOKEN_COMMENT)); // Skip line
  m_mapStringToToken.insert(std::make_pair("obj_info", PLY_TOKEN_OBJINFO)); // Skip line
  // The main data description containers:
  m_mapStringToToken.insert(std::make_pair("element", PLY_TOKEN_ELEMENT));
  m_mapStringToToken.insert(std::make_pair("property", PLY_TOKEN_PROPERTY));
  // Official datatypes:
  m_mapStringToToken.insert(std::make_pair("int8", PLY_TOKEN_CHAR));
  m_mapStringToToken.insert(std::make_pair("uint8", PLY_TOKEN_UCHAR));
  m_mapStringToToken.insert(std::make_pair("int16", PLY_TOKEN_SHORT));
  m_mapStringToToken.insert(std::make_pair("uint16", PLY_TOKEN_USHORT));
  m_mapStringToToken.insert(std::make_pair("int32", PLY_TOKEN_INT));
  m_mapStringToToken.insert(std::make_pair("uint32", PLY_TOKEN_UINT));
  m_mapStringToToken.insert(std::make_pair("float32", PLY_TOKEN_FLOAT));
  m_mapStringToToken.insert(std::make_pair("float64", PLY_TOKEN_DOUBLE));
  // Datatypes found in old files:
  m_mapStringToToken.insert(std::make_pair("char", PLY_TOKEN_CHAR));
  m_mapStringToToken.insert(std::make_pair("uchar", PLY_TOKEN_UCHAR));
  m_mapStringToToken.insert(std::make_pair("short", PLY_TOKEN_SHORT));
  m_mapStringToToken.insert(std::make_pair("ushort", PLY_TOKEN_USHORT));
  m_mapStringToToken.insert(std::make_pair("int", PLY_TOKEN_INT));
  m_mapStringToToken.insert(std::make_pair("uint", PLY_TOKEN_UINT));
  m_mapStringToToken.insert(std::make_pair("float", PLY_TOKEN_FLOAT));
  m_mapStringToToken.insert(std::make_pair("double", PLY_TOKEN_DOUBLE));
  // List type property:
  m_mapStringToToken.insert(std::make_pair("list", PLY_TOKEN_LIST));
  // End of header magic:
  m_mapStringToToken.insert(std::make_pair("end_header", PLY_TOKEN_ENDHEADER));

  // Map known user defined names to PLY_ATTRIBUTE_COMPONENT IDs for simpler code.
  m_mapStringToAttributeComponent.insert(std::make_pair("x", PLY_VERTEX_X)); // Vertex.xyz
  m_mapStringToAttributeComponent.insert(std::make_pair("y", PLY_VERTEX_Y));
  m_mapStringToAttributeComponent.insert(std::make_pair("z", PLY_VERTEX_Z));
  m_mapStringToAttributeComponent.insert(std::make_pair("nx", PLY_NORMAL_X)); // Normal.xyz
  m_mapStringToAttributeComponent.insert(std::make_pair("ny", PLY_NORMAL_Y));
  m_mapStringToAttributeComponent.insert(std::make_pair("nz", PLY_NORMAL_Z));
  m_mapStringToAttributeComponent.insert(std::make_pair("red",   PLY_COLOR_R)); // Color.rgb
  m_mapStringToAttributeComponent.insert(std::make_pair("green", PLY_COLOR_G));
  m_mapStringToAttributeComponent.insert(std::make_pair("blue",  PLY_COLOR_B));
  //m_mapStringToAttributeComponent.insert(std::make_pair("s", PLY_TOKEN_S)); // TexCoord // Not seen, yet.
  //m_mapStringToAttributeComponent.insert(std::make_pair("t", PLY_TOKEN_T));
}


// TODO: Extend the plug-in interface callbacks to handle an onError message.
void  PLYLoader::onError(const string &message) const
{
  if ( callback() )
  {
    callback()->onError(PlugInCallback::PICE_UNSPECIFIED_ERROR, NULL);
  }
}

bool  PLYLoader::onIncompatibleValues( int value0, int value1, const string &node, const string &field0, const string &field1 ) const
{
  return( callback() ? callback()->onIncompatibleValues( m_line, node, field0, value0, field1, value1) : true );
}

template<typename T> bool  PLYLoader::onInvalidValue( T value, const string &node, const string &field ) const
{
  return( callback() ? callback()->onInvalidValue( m_line, node, field, value ) : true );
}

bool  PLYLoader::onEmptyToken( const string &tokenType, const string &token ) const
{
  return( callback() ? callback()->onEmptyToken( m_line, tokenType, token ) : true );
}

bool  PLYLoader::onFileNotFound( const string &file ) const
{
  return( callback() ? callback()->onFileNotFound( file ) : true );
}

bool  PLYLoader::onFilesNotFound( bool found, const vector<string> &files ) const
{
  return( callback() && !found ? callback()->onFilesNotFound( files ) : true );
}

void  PLYLoader::onUnexpectedEndOfFile( bool error ) const
{
  if ( callback() && error )
  {
    callback()->onUnexpectedEndOfFile( m_line );
  }
}

void  PLYLoader::onUnexpectedToken( const string &expected, const string &token ) const
{
  if ( callback() && ( expected != token ) )
  {
    callback()->onUnexpectedToken( m_line, expected, token );
  }
}

void  PLYLoader::onUnknownToken( const string &context, const string &token ) const
{
  if ( callback() )
  {
    callback()->onUnknownToken( m_line, context, token );
  }
}

bool  PLYLoader::onUndefinedToken( const string &context, const string &token ) const
{
  return( callback() ? callback()->onUndefinedToken( m_line, context, token ) : true );
}

bool  PLYLoader::onUnsupportedToken( const string &context, const string &token ) const
{
  return( callback() ? callback()->onUnsupportedToken( m_line, context, token ) : true );
}





// ########## Read functions



// char 8-bits
void PLYLoader::readAscii_CHAR(void *dst)
{
  int advance = lookAheadToken();
  *(char *) dst = (char) atoi(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readAnyEndian_CHAR(void *dst)
{
  *(char *) dst = *m_pcCurrent;
  m_pcCurrent++;
}

// unsigned char 8-bits
void PLYLoader::readAscii_UCHAR(void *dst)
{
  int advance = lookAheadToken();
  *(unsigned char *) dst = (unsigned char) atoi(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readAnyEndian_UCHAR(void *dst)
{
  *(unsigned char *) dst = *(unsigned char *) m_pcCurrent;
  m_pcCurrent++;
}

// short 16-bits
void PLYLoader::readAscii_SHORT(void *dst)
{
  int advance = lookAheadToken();
  *(short *) dst = (short) atoi(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readLittleEndian_SHORT(void *dst)
{
  *(short *) dst = *(short *) m_pcCurrent;
  m_pcCurrent += 2;
}

void PLYLoader::readBigEndian_SHORT(void *dst)
{
  char c[2];

  c[1] = m_pcCurrent[0];
  c[0] = m_pcCurrent[1];
  *(short *) dst = *(short *) c;
  m_pcCurrent += 2;
}

// unsigned short 16-bits
void PLYLoader::readAscii_USHORT(void *dst)
{
  int advance = lookAheadToken();
  *(unsigned short *) dst = (unsigned short) atoi(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readLittleEndian_USHORT(void *dst)
{
  *(unsigned short *) dst = *(unsigned short *) m_pcCurrent;
  m_pcCurrent += 2;
}

void PLYLoader::readBigEndian_USHORT(void *dst)
{
  char c[2];

  c[1] = m_pcCurrent[0];
  c[0] = m_pcCurrent[1];
  *(unsigned short *) dst = *(unsigned short *) c;
  m_pcCurrent += 2;
}

// int 32-bits
void PLYLoader::readAscii_INT(void *dst)
{
  int advance = lookAheadToken();
  *(int *) dst = atoi(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readLittleEndian_INT(void *dst)
{
  *(int *) dst = *(int *) m_pcCurrent;
  m_pcCurrent += 4;
}

void PLYLoader::readBigEndian_INT(void *dst)
{
  char c[4];

  c[3] = m_pcCurrent[0];
  c[2] = m_pcCurrent[1];
  c[1] = m_pcCurrent[2];
  c[0] = m_pcCurrent[3];
  *(int *) dst = *(int *) c;
  m_pcCurrent += 4;
}

// unsigned int 32-bits
void PLYLoader::readAscii_UINT(void *dst)
{
  int advance = lookAheadToken();
  *(unsigned int *) dst = (unsigned int) atoi(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readLittleEndian_UINT(void *dst)
{
  *(unsigned int *) dst = *(unsigned int *) m_pcCurrent;
  m_pcCurrent += 4;
}

void PLYLoader::readBigEndian_UINT(void *dst)
{
  char c[4];

  c[3] = m_pcCurrent[0];
  c[2] = m_pcCurrent[1];
  c[1] = m_pcCurrent[2];
  c[0] = m_pcCurrent[3];
  *(unsigned int *) dst = *(unsigned int *) c;
  m_pcCurrent += 4;
}

// float 32-bit
void PLYLoader::readAscii_FLOAT(void *dst)
{
  int advance = lookAheadToken();
  *(float *) dst = (float) atof(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readLittleEndian_FLOAT(void *dst)
{
  *(float *) dst = *(float *) m_pcCurrent;
  m_pcCurrent += 4;
}

void PLYLoader::readBigEndian_FLOAT(void *dst)
{
  char c[4];

  c[3] = m_pcCurrent[0];
  c[2] = m_pcCurrent[1];
  c[1] = m_pcCurrent[2];
  c[0] = m_pcCurrent[3];
  *(float *) dst = *(float *) c;
  m_pcCurrent += 4;
}


// double 64-bit
void PLYLoader::readAscii_DOUBLE(void *dst)
{
  int advance = lookAheadToken();
  *(double *) dst = (float) atof(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readLittleEndian_DOUBLE(void *dst)
{
  *(float *) dst = (float) (*(double *) m_pcCurrent);
  m_pcCurrent += 8;
}

void PLYLoader::readBigEndian_DOUBLE(void *dst)
{
  char c[8];

  c[7] = m_pcCurrent[0];
  c[6] = m_pcCurrent[1];
  c[5] = m_pcCurrent[2];
  c[4] = m_pcCurrent[3];
  c[3] = m_pcCurrent[4];
  c[2] = m_pcCurrent[5];
  c[1] = m_pcCurrent[6];
  c[0] = m_pcCurrent[7];
  *(float *) dst = (float) (*(double *) c);
  m_pcCurrent += 8;
}



// Functions to read attribute data!!!
// Almost the same as the functions above, except that they scale non-float data 
// to unit ranges and return float data throughout. Think of unsigned char colors!
// Conversion is applied according to the OpenGL 2.1 specs Table 2.9 "Component Conversions"

// char 8-bits
void PLYLoader::readAttributeAscii_CHAR(float *dst)
{
  int advance = lookAheadToken();
  float f = (float) (char) atoi(m_token);
  *dst = (2.0f * f + 1.0f) / 255.0f;
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeAnyEndian_CHAR(float *dst)
{
  float f = (float) *m_pcCurrent;
  *dst = (2.0f * f + 1.0f) / 255.0f;
  m_pcCurrent++;
}

// unsigned char 8-bits
void PLYLoader::readAttributeAscii_UCHAR(float *dst)
{
  int advance = lookAheadToken();
  float f = (float) (unsigned char) atoi(m_token);
  *dst = f / 255.0f;
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeAnyEndian_UCHAR(float *dst)
{
  float f = (float) (*(unsigned char *) m_pcCurrent);
  *dst = f / 255.0f;
  m_pcCurrent++;
}

// short 16-bits
void PLYLoader::readAttributeAscii_SHORT(float *dst)
{
  int advance = lookAheadToken();
  float f = (float) (short) atoi(m_token);
  *dst = (2.0f * f + 1.0f) / 65535.0f;
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeLittleEndian_SHORT(float *dst)
{
  float f = (float) (*(short *) m_pcCurrent);
  *dst = (2.0f * f + 1) / 65535.0f;
  m_pcCurrent += 2;
}

void PLYLoader::readAttributeBigEndian_SHORT(float *dst)
{
  char c[2];

  c[1] = m_pcCurrent[0];
  c[0] = m_pcCurrent[1];
  float f = (float) (*(short *) c);
  *dst = (2.0f * f + 1) / 65535.0f;
  m_pcCurrent += 2;
}

// unsigned short 16-bits
void PLYLoader::readAttributeAscii_USHORT(float *dst)
{
  int advance = lookAheadToken();
  float f = (float) (unsigned short) atoi(m_token);
  *dst = f / 65535.0f;
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeLittleEndian_USHORT(float *dst)
{
  float f = (float) (*(unsigned short *) m_pcCurrent);
  *dst = f / 65535.0f;
  m_pcCurrent += 2;
}

void PLYLoader::readAttributeBigEndian_USHORT(float *dst)
{
  char c[2];

  c[1] = m_pcCurrent[0];
  c[0] = m_pcCurrent[1];
  float f = (float) (*(unsigned short *) c);
  *dst = f / 65535.0f;
  m_pcCurrent += 2;
}

// int 32-bits
void PLYLoader::readAttributeAscii_INT(float *dst)
{
  int advance = lookAheadToken();
  double d = (double) atoi(m_token);
  *dst = (float) ((2.0 * d + 1.0) / 4294967295.0);
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeLittleEndian_INT(float *dst)
{
  double d = (double) (*(int *) m_pcCurrent);
  *dst = (float) ((2.0 * d + 1.0) / 4294967295.0);
  m_pcCurrent += 4;
}

void PLYLoader::readAttributeBigEndian_INT(float *dst)
{
  char c[4];

  c[3] = m_pcCurrent[0];
  c[2] = m_pcCurrent[1];
  c[1] = m_pcCurrent[2];
  c[0] = m_pcCurrent[3];
  double d = (double) (*(int *) c);
  *dst = (float) ((2.0 * d + 1.0) / 4294967295.0);
  m_pcCurrent += 4;
}

// unsigned int 32-bits
void PLYLoader::readAttributeAscii_UINT(float *dst)
{
  int advance = lookAheadToken();
  double d = (double) (unsigned int) atoi(m_token);
  *dst = (float) (d / 4294967295.0);
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeLittleEndian_UINT(float *dst)
{
  double d = (double) (*(unsigned int *) m_pcCurrent);
  *dst = (float) (d / 4294967295.0);
  m_pcCurrent += 4;
}

void PLYLoader::readAttributeBigEndian_UINT(float *dst)
{
  char c[4];

  c[3] = m_pcCurrent[0];
  c[2] = m_pcCurrent[1];
  c[1] = m_pcCurrent[2];
  c[0] = m_pcCurrent[3];
  double d = (double) (*(unsigned int *) c);
  *dst = (float) (d / 4294967295.0);
  m_pcCurrent += 4;
}

// float 32-bit
void PLYLoader::readAttributeAscii_FLOAT(float *dst)
{
  int advance = lookAheadToken();
  *dst = (float) atof(m_token);
  m_pcCurrent += advance;
}

void PLYLoader::readAttributeLittleEndian_FLOAT(float *dst)
{
  *dst = *(float *) m_pcCurrent;
  m_pcCurrent += 4;
}

void PLYLoader::readAttributeBigEndian_FLOAT(float *dst)
{
  char c[4];

  c[3] = m_pcCurrent[0];
  c[2] = m_pcCurrent[1];
  c[1] = m_pcCurrent[2];
  c[0] = m_pcCurrent[3];
  *dst = *(float *) c;
  m_pcCurrent += 4;
}


// double 64-bit
// void PLYLoader::readAttributeAscii_DOUBLE(float *dst) // Same as _FLOAT
//{
//  int advance = lookAheadToken();
//  *dst = (float) atof(m_token);
//  m_pcCurrent += advance;
//}

void PLYLoader::readAttributeLittleEndian_DOUBLE(float *dst)
{
  *dst = (float) (*(double *) m_pcCurrent);
  m_pcCurrent += 8;
}

void PLYLoader::readAttributeBigEndian_DOUBLE(float *dst)
{
  char c[8];

  c[7] = m_pcCurrent[0];
  c[6] = m_pcCurrent[1];
  c[5] = m_pcCurrent[2];
  c[4] = m_pcCurrent[3];
  c[3] = m_pcCurrent[4];
  c[2] = m_pcCurrent[5];
  c[1] = m_pcCurrent[6];
  c[0] = m_pcCurrent[7];
  *dst = (float) (*(double *) c);
  m_pcCurrent += 8;
}


// Don't write another 24 functions just to get an unsigned int returned.
// The switch looks costly though.
unsigned int PLYLoader::readListCounterOrIndex(PFN_READ pfn, PLY_TOKEN type)
{
  union
  {
    char c;
    unsigned char uc;
    short s;
    unsigned short us;
    int i;
    unsigned int ui;
    float f;
    double d;
  } data;

  (this->*pfn)(&data); // Read the data as is.

  unsigned int value = 0;

  // I have found multiple files which used signed data types (e.g. short) to build a list of indices.
  // Assume that if any of those are negative, it has been meant to be unsigned, so fall through on the signed ones and see if that helps
  switch (type)
  {
    case PLY_TOKEN_CHAR:
      if (data.c < 0)
      {
        onInvalidValue(data.c, "Expected positive list count or vertex index.", "char");
        return 0;
      }
      value = (unsigned int) data.c;
      break;
    case PLY_TOKEN_UCHAR:
      value = (unsigned int) data.uc;
      break;
    case PLY_TOKEN_SHORT:
      if (data.s < 0)
      {
        onInvalidValue(data.s, "Expected positive list count or vertex index", "short");
        return 0;
      }
      value = (unsigned int) data.s;
      break;
    case PLY_TOKEN_USHORT:
      value = (unsigned int) data.us;
      break;
    case PLY_TOKEN_INT:
      if (data.i < 0)
      {
        onInvalidValue(data.i, "Expected positive list count or vertex index.", "int");
        return 0;
      }
      value = (unsigned int) data.i;
      break;
    case PLY_TOKEN_UINT:
      value = (unsigned int) data.ui;
      break;
    case PLY_TOKEN_FLOAT:
      if (data.f < 0)
      {
        onInvalidValue(data.f, "Expected positive list count or vertex index.", "float");
        // Actually didn't expect float indices at all.
        return 0;
      }
      value = (unsigned int) data.f;
      break;
    case PLY_TOKEN_DOUBLE:
      if (data.d < 0)
      {
        onInvalidValue((float) data.d, "Expected positive list count or vertex index.", "double"); // float cast is needed for overload resolve.
        // Actually didn't expect double indices at all.
        return 0;
      }
      value = (unsigned int) data.d;
      break;
    default:
      DP_ASSERT(0);
      break;
  }

  return value;                   
}
