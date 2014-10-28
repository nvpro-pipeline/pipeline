// Copyright NVIDIA Corporation 2002-2007
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

#include <fstream>
#include <string>

#include <dp/sg/core/nvsgapi.h>
#include <dp/sg/core/CoreTypes.h>
#include <dp/util/PlugInCallback.h>
#include <dp/util/File.h>
#include <dp/util/FileMapping.h>
#include <dp/sg/io/PlugInterface.h>


// convenient aliases for used built-in types
typedef char                    byte_t;   //!< Specifies an 8-bit signed type.
typedef unsigned char           ubyte_t;  //!< Specifies an 8-bit unsigned type.
typedef int                     int_t;    //!< Specifies a 32-bit signed integer type.
typedef unsigned int            uint_t;   //!< Specifies a 32-bit unsigned integer type.
#ifdef LINUX
# if __WORDSIZE == 64           // defined indirectly through stdint.h
// avoid a conflict with GNU stdint.h on Linux64
// note: long is a 64-bit type on Linux64, while it is 32bit on Win64
// Linux64
typedef unsigned long           uint64_t; //!< Specifies a 64-bit unsigned integer type.
# else
// Linux32
typedef unsigned long long      uint64_t; //!< Specifies a 64-bit unsigned integer type.
# endif
#else
// Win32 and Win64
typedef unsigned long long      uint64_t; //!< Specifies a 64-bit unsigned integer type.
#endif




// storage-class defines 
#if ! defined( DOXYGEN_IGNORE )
# if defined(_WIN32)
#  ifdef PLYLOADER_EXPORTS
#   define PLYLOADER_API __declspec(dllexport)
#  else
#   define PLYLOADER_API __declspec(dllimport)
#  endif
# else
#   define PLYLOADER_API
# endif
#endif

#if defined(LINUX)
typedef unsigned int DWORD;
typedef bool BOOL;
void lib_init() __attribute__ ((constructor));   // will be called before dlopen() returns
#endif


// exports required for a scene loader plug-in
extern "C"
{
//! Get the PlugIn interface for this scene loader.
/** Every PlugIn has to resolve this function. It is used to get a pointer to a PlugIn class, in this case a PLYLoader.
  * If the PlugIn ID \a piid equals \c PIID_PLY_SCENE_LOADER, a PLYLoader is created and returned in \a pi.
  * \returns  true, if the requested PlugIn could be created, otherwise false
  */
PLYLOADER_API bool getPlugInterface(const dp::util::UPIID& piid, dp::util::SmartPlugIn & pi);

//! Query the supported types of PlugIn Interfaces.
PLYLOADER_API void queryPlugInterfacePIIDs( std::vector<dp::util::UPIID> & piids );
}


// PLY Datatypes:
// name        type        number of bytes   alias added in this loader:
//                                           (Often found in older files!)
// ---------------------------------------   -----------------------------
// int8       character                 1    char
// uint8      unsigned character        1    uchar
// int16      short integer             2    short
// uint16     unsigned short integer    2    ushort
// int32      integer                   4    int
// uint32     unsigned integer          4    uint
// float32    single-precision float    4    float
// float64    double-precision float    8    double


// These enums encode the predefined keywords inside standard PLY files.
typedef enum 
{
  PLY_TOKEN_CHAR = 0,   // These data type tokens are zero based and used to index read function tables.
  PLY_TOKEN_UCHAR,
  PLY_TOKEN_SHORT,
  PLY_TOKEN_USHORT,
  PLY_TOKEN_INT,
  PLY_TOKEN_UINT,
  PLY_TOKEN_FLOAT,
  PLY_TOKEN_DOUBLE,
  
  PLY_TOKEN_PLY,
  PLY_TOKEN_FORMAT,
  
  PLY_TOKEN_ASCII,                // These three enums must be consecutive in this order to be used for function table lookup.
  PLY_TOKEN_BINARYLITTLEENDIAN,   // "-"
  PLY_TOKEN_BINARYBIGENDIAN,      // "-"
  
  PLY_TOKEN_ONEPOINTZERO,
  PLY_TOKEN_COMMENT,
  PLY_TOKEN_OBJINFO,
  PLY_TOKEN_ELEMENT,
  PLY_TOKEN_PROPERTY,
  PLY_TOKEN_LIST,
  PLY_TOKEN_ENDHEADER,

  PLY_TOKEN_UNKNOWN

} PLY_TOKEN;


// These enums define the states of a simple state machine used during parsing of the PLY header.
typedef enum
{
  PLY_STATE_PLY,
  PLY_STATE_FORMAT,
  PLY_STATE_FORMAT_TYPE,
  PLY_STATE_FORMAT_VERSION,
  PLY_STATE_ELEMENT,
  PLY_STATE_ELEMENT_IDENTIFIER,
  PLY_STATE_ELEMENT_COUNT,
  PLY_STATE_PROPERTY,
  PLY_STATE_PROPERTY_TYPE,
  PLY_STATE_PROPERTY_LIST_COUNT_TYPE,
  PLY_STATE_PROPERTY_LIST_DATA_TYPE,
  PLY_STATE_PROPERTY_NAME,
  PLY_STATE_END
} PLY_PARSER_STATE;


typedef enum 
{
  PLY_VERTEX_X = 0, // Zero based, used as indices and attribute bit flag location.
  PLY_VERTEX_Y,
  PLY_VERTEX_Z,
  PLY_NORMAL_X,
  PLY_NORMAL_Y,
  PLY_NORMAL_Z,
  PLY_COLOR_R,
  PLY_COLOR_G,
  PLY_COLOR_B,
  PLY_USER_DEFINED_COMPONENT // Will be ignored.
} PLY_ATTRIBUTE_COMPONENT;


#define ATTRIBUTE_MASK_VERTEX  ((1 << PLY_VERTEX_X) | (1 << PLY_VERTEX_Y) | (1 << PLY_VERTEX_Z))
#define ATTRIBUTE_MASK_NORMAL  ((1 << PLY_NORMAL_X) | (1 << PLY_NORMAL_Y) | (1 << PLY_NORMAL_Z))
#define ATTRIBUTE_MASK_COLOR   ((1 << PLY_COLOR_R)  | (1 << PLY_COLOR_G)  | (1 << PLY_COLOR_B))


class PLYLoader;

// Generic function pointer to write data from binary input data starting at m_pcCurrent to dst address.
typedef void (PLYLoader::*PFN_READ)(void *dst);
// Similar to above but data conversion to float based on OpenGL component conversion rules (OpenGL 2.1 specs Table 2.9)
typedef void (PLYLoader::*PFN_READ_ATTRIBUTE)(float *dst);


class PLYProperty
{
public:
    PLYProperty();
    ~PLYProperty();

    std::string name;      // Name of this poperty inside this element. Required to identify attributes (so far x, y, z, nx, ny, nz).
    PLY_TOKEN countType;   // Scalar integer count for list properties. TOKEN_UNKNOWN indicates non-list property.
    PLY_TOKEN dataType;    // Scalar data type for any property.
    PFN_READ pfnReadCount; // Read function for this property count, set if it's a list.
    PFN_READ pfnReadData;  // Read function for this property data, set if it's a list.
    PFN_READ_ATTRIBUTE pfnReadAttribute;  // Read function for this property data, set if it's an attribute. (Automatic conversion!)
    unsigned int index;    // Index inside attribute float array, the last element outside the supported attribute components is used to ignore unknown attribute data.
};


class PLYElement
{
public:
  PLYElement();
  ~PLYElement();

  std::string name;                           // Name of this element.
  unsigned int count;                         // Number of elements of this type in the file.
  std::vector<PLYProperty *> m_pProperties;   // Variable number of property fields per element.
};

SMART_TYPES( PLYLoader );

//! A Scene Loader for PLY files.
/** PLY are simple single geometry files often used in academia.
 ** Most famous scenes found in PLY format are the Stanford Bunny, Happy Buddha, Dragon, and Armadillo.
 ** A repository of model can be found here: 
 */
class PLYLoader : public dp::sg::io::SceneLoader
{
  public :
    static SmartPLYLoader create();
    ~PLYLoader();

  protected:
    PLYLoader();

    //! Maps \a numBytes bytes at file offset \a offset into process memory. 
    /** This function turns a given offset into a pointer and ensures that a minimum of \a numBytes bytes are mapped.
    * \returns A pointer to the mapped memory. */
    ubyte_t * mapOffset( uint_t offset    //!< File offset of the memory block to map.
                       , size_t numBytes  //!< Amount of bytes to map into process memory.
                       );

    //! Unmaps the memory that previously was mapped through mapOffset from the process' address space.
    /** The function accepts a pointer to the mapped file offset that previously was received by a call to mapOffset. */
    void unmapOffset(ubyte_t * offsetPtr  //!< Address where the file offset was mapped.
                    );

  public :
    //! Realization of the pure virtual interface function of a SceneLoader.
    /** Loads a PLY file given by \a filename. It looks for this file and 
      * possibly referenced other files like textures or effects at the given 
      * path first, then at the current location and finally it searches
      * through the \a searchPaths.
      * \returns  A pointer to the loaded scene. */
    dp::sg::core::SceneSharedPtr load( std::string const& filename                  //!<  file to load
                                     , std::vector<std::string> const& searchPaths  //!<  paths to search through
                                     , dp::sg::ui::ViewStateSharedPtr & viewState   /*!< If the function succeeded, this points to the optional
                                                                                         ViewState stored with the scene. */
                                     );

  private :

    //! An auxiliary helper template class which provides exception safe mapping and unmapping of file offsets.
    /** The purpose of this template class is to turn a mapped offset into an exception safe auto object, 
      * that is - the mapped offset automatically gets unmapped if the object runs out of scope. */
    template<typename T>
    class Offset_AutoPtr
    {
      public:
        //! Maps the specified file offset into process memory.
        /** This constructor is called on instantiation. 
        * It maps \a count objects of type T at file offset \a offset into process memory. */
        Offset_AutoPtr( dp::util::ReadMapping * fm, dp::util::SmartPlugInCallback const& pic, 
                        uint_t offset, size_t count = 1 );

        //! Unmaps the bytes, that have been mapped at instantiation, from process memory. 
        ~Offset_AutoPtr();

        //! Provides pointer-like access to the dumb pointer. 
        T* operator->() const;

        //! De-references the dumb pointer. 
        T& operator*() const;

        //! Implicit conversion to const T*. 
        operator const T*() const;

        //! Resets the object to map another file offset
        /** The function first unmaps previously mapped bytes and after that
        * maps \a count T objects at file offset \a offset into process memory. */
        void reset( uint_t offset, size_t count = 1 );

      private:
        T                             * m_ptr;
        dp::util::SmartPlugInCallback   m_pic;
        dp::util::ReadMapping         * m_fm;
    };

    // Parsing a binary filemapping.
    dp::util::ReadMapping * m_fm;

    std::map<std::string, PLY_TOKEN>               m_mapStringToToken;
    std::map<std::string, PLY_ATTRIBUTE_COMPONENT> m_mapStringToAttributeComponent;

    char *m_pcCurrent;
    char *m_pcEOF;
    char m_token[256];
    int m_plyFormat;  
    std::vector<PLYElement *> m_pElements;

    
    void cleanup(void);
    void initializeMapStringToToken(void);
    int lookAheadToken(void);
    int skipLine(void);

    void readAttributeAscii_CHAR(float *dst);  
    void readAttributeAnyEndian_CHAR(float *dst);
    void readAttributeAscii_UCHAR(float *dst);
    void readAttributeAnyEndian_UCHAR(float *dst);
    void readAttributeAscii_SHORT(float *dst);
    void readAttributeLittleEndian_SHORT(float *dst);
    void readAttributeBigEndian_SHORT(float *dst);
    void readAttributeAscii_USHORT(float *dst);
    void readAttributeLittleEndian_USHORT(float *dst);
    void readAttributeBigEndian_USHORT(float *dst);
    void readAttributeAscii_INT(float *dst);
    void readAttributeLittleEndian_INT(float *dst);
    void readAttributeBigEndian_INT(float *dst);
    void readAttributeAscii_UINT(float *dst);
    void readAttributeLittleEndian_UINT(float *dst);
    void readAttributeBigEndian_UINT(float *dst);
    void readAttributeAscii_FLOAT(float *dst);
    void readAttributeLittleEndian_FLOAT(float *dst);
    void readAttributeBigEndian_FLOAT(float *dst);
    void readAttributeLittleEndian_DOUBLE(float *dst);
    void readAttributeBigEndian_DOUBLE(float *dst);

    void readAscii_CHAR(void *dst);  
    void readAnyEndian_CHAR(void *dst);
    void readAscii_UCHAR(void *dst);
    void readAnyEndian_UCHAR(void *dst);
    void readAscii_SHORT(void *dst);
    void readLittleEndian_SHORT(void *dst);
    void readBigEndian_SHORT(void *dst);
    void readAscii_USHORT(void *dst);
    void readLittleEndian_USHORT(void *dst);
    void readBigEndian_USHORT(void *dst);
    void readAscii_INT(void *dst);
    void readLittleEndian_INT(void *dst);
    void readBigEndian_INT(void *dst);
    void readAscii_UINT(void *dst);
    void readLittleEndian_UINT(void *dst);
    void readBigEndian_UINT(void *dst);
    void readAscii_FLOAT(void *dst);
    void readLittleEndian_FLOAT(void *dst);
    void readBigEndian_FLOAT(void *dst);
    void readAscii_DOUBLE(void *dst);
    void readLittleEndian_DOUBLE(void *dst);
    void readBigEndian_DOUBLE(void *dst);

    unsigned int readListCounterOrIndex(PFN_READ pfn, PLY_TOKEN type);
    void ignoreProperty(PLYProperty *p);

    // Tables of read functions.
    PFN_READ_ATTRIBUTE m_apfnReadAttribute[8][3]; // Special in that it converts integer data to float according to the OpenGL specs.
    PFN_READ           m_apfnRead[8][3];          // Read data as it is. Caller needs to figure out what it was.


    void                       onError(const std::string &message) const;
    bool                       onIncompatibleValues( int value0, int value1, const std::string &node, const std::string &field0, const std::string &field1 ) const;
    template<typename T> bool  onInvalidValue( T value, const std::string &node, const std::string &field ) const;
    bool                       onEmptyToken( const std::string &tokenType, const std::string &token ) const;
    bool                       onFileNotFound( const std::string &file ) const;
    bool                       onFilesNotFound( bool found, const std::vector<std::string> &files ) const;
    void                       onUnexpectedEndOfFile( bool error ) const;
    void                       onUnexpectedToken( const std::string &expected, const std::string &token ) const;
    void                       onUnknownToken( const std::string &context, const std::string &token ) const;
    bool                       onUndefinedToken( const std::string &context, const std::string &token ) const;
    bool                       onUnsupportedToken( const std::string &context, const std::string &token ) const;

  private :

    unsigned int                       m_line;
    std::vector<std::string>           m_searchPaths;
};


inline ubyte_t * PLYLoader::mapOffset( uint_t offset, size_t numBytes )
{
  DP_ASSERT( m_fm );
  return( (ubyte_t*) m_fm->mapIn( offset, numBytes ) );
}

inline void PLYLoader::unmapOffset( ubyte_t * offsetPtr )
{
  DP_ASSERT( m_fm );
  m_fm->mapOut( offsetPtr );
}

template<typename T>
inline PLYLoader::Offset_AutoPtr<T>::Offset_AutoPtr( dp::util::ReadMapping * fm
                                                   , dp::util::SmartPlugInCallback const& pic
                                                   , uint_t offset, size_t count )
: m_ptr(NULL)
, m_fm(fm)
, m_pic(pic)
{
  if ( count )
  {
    m_ptr = (T*)m_fm->mapIn(offset, count*sizeof(T));
    if ( ! m_ptr && m_pic )
    {
      m_pic->onFileMappingFailed(m_fm->getLastError());
    }
  }
 }

template<typename T>
inline PLYLoader::Offset_AutoPtr<T>::~Offset_AutoPtr()
{
  if ( m_ptr )
  {
    m_fm->mapOut((ubyte_t*)m_ptr);
  }
}

template<typename T>
inline T* PLYLoader::Offset_AutoPtr<T>::operator->() const
{
  return m_ptr;
}

template<typename T>
inline T& PLYLoader::Offset_AutoPtr<T>::operator*() const
{
  return *m_ptr;
}

template<typename T>
inline PLYLoader::Offset_AutoPtr<T>::operator const T*() const
{
  return m_ptr;
}

template<typename T>
inline void PLYLoader::Offset_AutoPtr<T>::reset( uint_t offset, size_t count )
{
  if ( m_ptr )
  {
    m_fm->mapOut((ubyte_t*)m_ptr);
    m_ptr=NULL;
  }
  if ( count )
  {
    m_ptr = (T*)m_fm->mapIn(offset, count*sizeof(T));
    if ( ! m_ptr && m_pic )
    {
      m_pic->onFileMappingFailed(m_fm->getLastError());
    }
  }
}
