// Copyright (c) 2002-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <string>
#include <vector>

namespace dp
{
  namespace util
  {
    DEFINE_PTR_TYPES( PlugInCallback );

    //! PlugInCallback base class
    /** A PlugInCallback object can be used to report warnings and errors that happen while using a PlugIn. It is applied
      * to a PlugIn via PlugIn::setCallback().
      * A PlugInCallback object consists of a set of virtual functions that can be overloaded by an application specific
      * PlugInCallback object.
      * It has two generic callbacks \a onError and \a onWarning, that can be called with any error/warning code.
      * And it has two families of specific error and warning functions that are called on specific error/warning
      * conditions.
      */
    class PlugInCallback
    {
      public:
        //! Enumeration of PlugInCallback errors.
        enum class Error
        {
          UNEXPECTED_EOF,
          UNEXPECTED_TOKEN,
          UNKNOWN_TOKEN,
          FILE_ACCESS_FAILED,
          FILE_MAPPING_FAILED,
          INCOMPATIBLE_FILE,
          INVALID_FILE,
          UNSPECIFIED_ERROR
        };

        //! Enumeration of PlugInCallback warnings.
        enum class Warning
        {
          FILE_EMPTY,
          FILE_NOT_FOUND,
          FILES_NOT_FOUND,
          EMPTY_TOKEN,
          INCOMPATIBLE_VALUES,
          INVALID_VALUE,
          UNDEFINED_TOKEN,
          UNSUPPORTED_TOKEN,
          DEGENERATE_GEOMETRY,
          UNSPECIFIED_WARNING
        };

        //! Enumeration of value types used in warnings/errors.
        enum class TypeID
        {
          INT,
          FLOAT
        };

        //! Information structure used for warning PICW_EMPTY_TOKEN
        typedef struct
        {
          unsigned int  position;           //!<  position in file, where the warning was raised
          std::string  context;            //!<  context of the warning
          std::string  token;              //!<  name of empty token
        } EmptyTokenInfo;

        //! Information structure used for warning PICW_INCOMPATIBLE_VALUES
        typedef struct
        {
          unsigned int  position;     //!<  position in file, where the warning was raised
          std::string   context;      //!<  context of the warning
          TypeID        valueType;    //!<  type of the incompatible values
          std::string   value0Name;   //!<  name of the first value
          const void *  value0;       //!<  pointer to the first value
          std::string   value1Name;   //!<  name of the second value
          const void *  value1;       //!<  pointer to the second value
        } IncompatibleValueInfo;

        //! Information structure used for warning PICW_INVALID_VALUE
        typedef struct
        {
          unsigned int  position;     //!<  position in file, where the warning was raised
          std::string   context;      //!<  context of the warning
          TypeID        valueType;    //!<  type of the invalid value
          std::string   valueName;    //!<  name of the value
          const void *  value;        //!<  pointer to the value
        } InvalidValueInfo;

        //! Information structure used for warning PICW_UNDEFINED_TOKEN
        typedef struct
        {
          unsigned int        position;     //!<  position in file, where the warning was raised
          std::string        context;      //!<  context of the undefined token
          std::string        token;        //!<  undefined token
        } UndefinedTokenInfo;

        //! Information structure used for error PICE_UNEXPECTED_TOKEN 
        typedef struct
        {
          unsigned int        position;     //!<  position in file, where the error was raised
          std::string        expected;     //!<  expected token
          std::string        encountered;  //!<  encountered token
        } UnexpectedTokenInfo;

        //! Information structure used for error PICE_UNKNOWN_TOKEN
        typedef struct
        {
          unsigned int        position;     //!<  position in file, where the error was raised
          std::string        context;      //!<  context of the unknown token
          std::string        token;        //!<  unknown token
        } UnknownTokenInfo;

        //! Information structure used for warning PICW_UNSUPPORTED_TOKEN
        typedef struct
        {
          unsigned int        position;     //!<  position in file, where the warning was raised
          std::string        context;      //!<  context of the unsupported token
          std::string        token;        //!<  unsupported token
        } UnsupportedTokenInfo;

        //! Information structure used for warning PICW_DEGENERATE_GEOMETRY
        typedef struct
        {
          unsigned int        position;   //!<  position in file, where the warning was raised
          std::string   name;       //!<  name of the geometry that was degenerate
        } DegenerateGeometryInfo;

        //! Information structure used for error PICE_FILE_ACCESS_ERROR
        typedef struct
        {
          std::string        file;         //!< Specifies the name of the file for which the access error occurred.
          unsigned int  systemSpecificErrorCode; //!< Specifies a system specific error code. 
        } FileAccessFailedInfo;

        //! Information structure used for error PICE_FILE_MAPPING_ERROR
        typedef struct
        {
          unsigned int  systemSpecificErrorCode; //!< Specifies a system specific error code. 
        } FileMappingFailedInfo;

        //! Information structure used for error PICE_INCOMPATIBLE_FILE
        typedef struct
        {
          std::string        file;         //!< Specifies the name of the file that was detected to be incompatible.
          std::string        context;      //!< Specifies the context of detected incompatibility.
          unsigned int  expectedVersion; //!< Specifies the expected version. The high-order 16-bits specify
                                         //!< the major version; the low-order 16-bits specify the compatibility level.   
          unsigned int  detectedVersion; //!< Specifies the detected version. The high-order 16-bits specify
                                         //!< the major version; the low-order 16-bits specify the compatibility level.   
        } IncompatibleFileInfo;      

        //! Information structure used for error PICE_INVALID_FILE
        typedef struct
        {
          std::string        file;         //!< Specifies the name of the file the was detected to be invalid.
          std::string        context;      //!< Specifies the context of detected invalidity.
        } InvalidFileInfo;

      public:
        static PlugInCallbackSharedPtr create();
        virtual ~PlugInCallback();

        //! Set whether an exception should be thrown on error.
        void  setThrowExceptionOnError( bool set );

        //! General callback on error.
        /** This general error callback is called with every error that isn't completely handled in a specialized
          * error callback.
          * If throwExceptionOnError is set (default), onError throws an PIC_ERROR exception.
          * The specific error callbacks fall back to this function.
          * This callback can be overloaded to support new error conditions.  */
        virtual void onError( Error eid           //!<  error ID
                            , const void *info    //!<  pointer to information structure corresponding to \a eid
                            ) const;

        //! General callback on warning.
        /** This general warning callback is called with every warning that isn't completely handled in a specialized
          * warning callback.
          * In this base implemetation, onWarning just returns true, meaning the PlugIn operation can be continued
          * without danger.
          * The specific warning callbacks fall back to this function.
          * This callback can be overloaded to support new warning conditions.  */
        virtual bool onWarning( Warning wid       //!<  waringing ID
                              , const void *info  //!<  pointer to information structure corresponding to \a wid
                              ) const;

        //! General callback to display an unlocalized message.
        /** This general callback is *DISCOURAGED* but can be used by a loader
          * in the event that an unstructured or unlocalized 
          * message must be presented to the user.
          *
          * 'severity' should contain either "WARNING" or "ERROR"
          * 'message' should contain the unlocalized text message
          * 
          * In this base implemetation, if the string severity == "WARNING" then
          * the function will return true; if the string severity == "ERROR" then
          * if throwExceptionOnError is set (default), throws PIC_ERROR exception.
          *
          * This callback can be overloaded to support new warning conditions.  */
        virtual bool onUnLocalizedMessage( const std::string & severity, 
                                           const std::string & message ) const;

        //! Specialized callback on error: PICE_UNEXPECTED_EOF
        /** This error callback is called, when the end of a file was reached unexpectedly.
          * In this base implementation, the general callback onError is called.
          * Overload this callback to implement special handling on unexpected end of file errors. */
        virtual void onUnexpectedEndOfFile( unsigned int position //!<  position in file, where error occured
                                          ) const;

        //! Specialized callback on error: PICE_UNEXPECTED_TOKEN
        /** This error callback is called, when an unexpected token is encountered.
          * In this base implementation, a struct UnexpectedTokenInfo is constructed out of the arguments, and the
          * general callback onError is called.
          * Overlaod this callback to implement special handling on unexpected token errors. */
        virtual void onUnexpectedToken( unsigned int position           //!<  position in file, where error occured
                                      , const std::string &expected    //!<  expected token
                                      , const std::string &encountered //!<  encountered token
                                      ) const;

        //! Specialized callback on error: PICE_UNKNOWN_TOKEN
        /** This error callback is called, when an unknown token is encountered.
          * In this base implementation, a struct UnknownTokenInfo is constructed out of the arguments, and the
          * general callback onError is called.
          * Overload this callback to implement special handling on unknown token errors. */
        virtual void onUnknownToken( unsigned int position        //!<  position in file, where the error occured
                                   , const std::string &context  //!<  context of the unknown token
                                   , const std::string &token    //!<  unknown token
                                   ) const;

        //! Specialized callback on warning: PICW_EMPTY_TOKEN
        /** This warning callback is called, when a token proved to be empty.
          * In this base implementation, a struct EmptyTokenInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on empty token warnings. */
        virtual bool onEmptyToken( unsigned int position        //!<  position in file, where the warning occured
                                 , const std::string &context  //!<  context of the empty token
                                 , const std::string &token    //!<  name of the empty token
                                 ) const;

        //! Specialized callback on warning: PICW_FILE_EMPTY
        /** This warning callback is called, when a file proved to be empty.
          * In this base implementation, the general callback onWarning is called.
          * Overload this callback to implement special handling on empty file warnings.  */
        virtual bool onFileEmpty( const std::string &file  //!<  name of the empty file
                                ) const;

        //! Specialized callback on warning: PICW_FILE_NOT_FOUND
        /** This warning callback is called, when a file wasn't found.
          * In this base implementation, the general callback onWarning is called.
          * Overload this callback to implement special handling on file not found warnings.  */
        virtual bool onFileNotFound( const std::string &file //!<  name of the file, that wasn't found
                                   ) const;

        //! Specialized callback on warning: PICW_FILES_NOT_FOUND
        /** This warning callback is called, when none of a number of files was found.
          * In this base implementation, the general callback onWarning is called.
          * Overload this callback to implement special handling on file not found warnings.  */
        virtual bool onFilesNotFound( const std::vector<std::string> &files //!<  vector of files, where none was found
                                    ) const;

        //! Specialized callback on warning: PICW_UNDEFINED_TOKEN
        /** This warning callback is called, a named token is used that wasn't defined before.
          * In this base implementation, a struct UndefinedTokenInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on undefined token warnings. */
        virtual bool onUndefinedToken( unsigned int position        //!<  position in file, where the warning occured
                                     , const std::string &context  //!<  context of the undefined token
                                     , const std::string &token    //!<  undefined token
                                     ) const;

        //! Specialized callback on warning: PICW_INCOMPATIBLE_VALUES
        /** This warning callback is called, when two incompatible values of type int are found in the file.
          * In this base implementation, a struct IncompatibleValueInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on incompatible values warnings.  */
        virtual bool onIncompatibleValues( unsigned int position          //!<  position in file, where the warning occured
                                         , const std::string &context    //!<  context of the incompatible values
                                         , const std::string &value0Name //!<  name of the first value
                                         , int value0               //!<  first value
                                         , const std::string &value1Name //!<  name of the second value
                                         , int value1               //!<  second value
                                         ) const;

        //! Specialized callback on warning: PICW_INCOMPATIBLE_VALUES
        /** This warning callback is called, when two incompatible values of type float are found in the file.
          * In this base implementation, a struct IncompatibleValueInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on incompatible values warnings.  */
        virtual bool onIncompatibleValues( unsigned int position          //!<  position in file, where the warning occured
                                         , const std::string &context    //!<  context of the incompatible values
                                         , const std::string &value0Name //!<  name of the first value
                                         , float value0             //!<  first value
                                         , const std::string &value1Name //!<  name of the second value
                                         , float value1             //!<  second value
                                         ) const;

        //! Specialized callback on warning: PICW_INVALID_VALUE
        /** This warning callback is called, when an invalid value of type int is found in the file.
          * In this base implementation, a struct InvalidValueInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on incompatible values warnings.  */
        virtual bool onInvalidValue( unsigned int position          //!<  position in file, where the warning occured
                                   , const std::string &context    //!<  context of the invalid value
                                   , const std::string &valueName  //!<  name of the invalid value
                                   , int value                //!<  invalid value
                                   ) const;

        //! Specialized callback on warning: PICW_INVALID_VALUE
        /** This warning callback is called, when an invalid value of type float is found in the file.
          * In this base implementation, a struct InvalidValueInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on incompatible values warnings.  */
        virtual bool onInvalidValue( unsigned int position          //!<  position in file, where the warning occured
                                   , const std::string &context    //!<  context of the invalid value
                                   , const std::string &valueName  //!<  name of the invalid value
                                   , float value              //!<  invalid value
                                   ) const;

        //! Specialized callback on warning: PICW_UNSUPPORTED_TOKEN
        /** This warning callback is called, when an unsupported token is found in the file.
          * In this base implementation, a struct UnsupportedTokenInfo is constructed out of the arguments, and the
          * general callback onWarning is called.
          * Overload this callback to implement special handling on unsupported token warnings.  */
        virtual bool onUnsupportedToken( unsigned int position        //!<  position in file, where the warning occured
                                       , const std::string &context  //!<  context of the unsupported token
                                       , const std::string &token    //!<  unsupported token
                                       ) const;

        //! Specialized callback on warning: PICW_DEGENERATE_GEOMETRY
        /** This warning callback is called when degenerate geometry is found in the file.
          * In this base implementation, a struct DegenerateGeometryInfo is constructed 
          * out of the arguments, and the general callback onWarning is called.
          * Overload this callback to implement special handling on unsupported token warnings.  */
        virtual bool onDegenerateGeometry( unsigned int position        //!<  position in file, where the warning occured
                                         , const std::string &context  //!<  name of geometry
                                         ) const;

        //! Error callback on PICE_FILE_ACCESS_FAILED error
        /** This callback should be invoked from within a plug-in if a file access error occured. 
          * The default implementation simply calls onError.\n
          * It is advisable to overload this function if more customized error handling is requested
          * for file access errors. */
        virtual void onFileAccessFailed( const std::string& file  //!< Specifies the name of the file
                                       , unsigned int systemSpecificErrorCode //!< Specifies a system specific error code.
                                       ) const;

        //! Error callback on PICE_FILE_MAPPING_FAILED error
        /** This callback should be invoked from within a plug-in if a file mapping error occured. 
          * The default implementation simply calls onError.\n
          * It is advisable to overload this function if more customized error handling is requested
          * for file mapping errors. */
        virtual void onFileMappingFailed( unsigned int systemSpecificErrorCode //!< Specifies a system specific error code.
                                        ) const; 

        //! Error callback on PICE_INCOMPATIBLE_FILE error
        /** This callback should be invoked from within a plug-in if a file is detected to be incompatible. 
          * The default implementation simply calls onError.\n
          * It is advisable to overload this function if more customized error handling is requested
          * when detecting incompatible files. */
        virtual void onIncompatibleFile( const std::string& file //!< Specifies the name of the file.
                                       , const std::string& context //!< Specifies the context of detected incompatibility.
                                       , unsigned int expectedVersion //!< Specifies the expected version. The high-order 16-bits specify
                                                                      //!< the major version; the low-order 16-bits specify the compatibility level.
                                       , unsigned int detectedVersion //!< Specifies the detected version. The high-order 16-bits specify
                                                                      //!< the major version; the low-order 16-bits specify the compatibility level.
                                       ) const; 

        //! Error callback on PICE_INVALID_FILE error
        /** This callback should be invoked from within a plug-in if a file is detected to be invalid. 
          * The default implementation simply calls onError.\n
          * It is advisable to overload this function if more customized error handling is requested
          * when detecting invalid files. */
        virtual void onInvalidFile( const std::string& file //!< Specifies the name of the file.
                                  , const std::string& context //!< Specifies the context of detected invalidity.
                                  ) const;

      protected: 
        PlugInCallback();

      private:
        bool  m_throwExceptionOnError;
    };

    inline PlugInCallbackSharedPtr PlugInCallback::create()
    {
      return( std::shared_ptr<PlugInCallback>( new PlugInCallback() ) );
    }

    inline  PlugInCallback::PlugInCallback()
      : m_throwExceptionOnError(true)
    {
    }

    inline  PlugInCallback::~PlugInCallback()
    {
    }

    inline  void  PlugInCallback::setThrowExceptionOnError( bool set )
    {
      m_throwExceptionOnError = set;
    }

    inline  void  PlugInCallback::onError( Error eid, const void *info ) const
    {
      if ( m_throwExceptionOnError )
      {
        throw( eid );
      }
    }

    inline  bool  PlugInCallback::onWarning( Warning wid, const void *info ) const
    {
      return( true );
    }

    inline  bool  PlugInCallback::onUnLocalizedMessage(const std::string & severity,
                                                       const std::string & message )
                                                    const
    {
      if ( (severity == "ERROR") && m_throwExceptionOnError )
      {
        throw( Error::UNSPECIFIED_ERROR );
      }
      else
      {
        return( true );
      }
    }

    inline  void  PlugInCallback::onUnexpectedEndOfFile( unsigned int position ) const
    {
      onError( Error::UNEXPECTED_EOF, &position );
    }

    inline  void  PlugInCallback::onUnexpectedToken( unsigned int position, const std::string &expected, const std::string &encountered ) const
    {
      UnexpectedTokenInfo uti;
      uti.position    = position;
      uti.expected    = expected;
      uti.encountered = encountered;
      onError( Error::UNEXPECTED_TOKEN, &uti );
    }

    inline  void  PlugInCallback::onUnknownToken( unsigned int position, const std::string &context, const std::string &token ) const
    {
      UnknownTokenInfo  uti;
      uti.position  = position;
      uti.context   = context;
      uti.token     = token;
      onError( Error::UNKNOWN_TOKEN, &uti );
    }


    inline  bool  PlugInCallback::onEmptyToken( unsigned int position, const std::string &context, const std::string &token ) const
    {
      EmptyTokenInfo  uti;
      uti.position  = position;
      uti.context   = context;
      uti.token     = token;
      return( onWarning( Warning::EMPTY_TOKEN, &uti ) );
    }

    inline  bool  PlugInCallback::onDegenerateGeometry( unsigned int position, const std::string &name ) const
    {
      DegenerateGeometryInfo  dg;
      dg.position  = position;
      dg.name      = name;
      return( onWarning( Warning::DEGENERATE_GEOMETRY, &dg ) );
    }

    inline  bool  PlugInCallback::onFileEmpty( const std::string &file ) const
    {
      return( onWarning( Warning::FILE_EMPTY, &file ) );
    }

    inline  bool  PlugInCallback::onFileNotFound( const std::string &file ) const
    {
      return( onWarning( Warning::FILE_NOT_FOUND, &file ) );
    }

    inline  bool  PlugInCallback::onFilesNotFound( const std::vector<std::string> &files ) const
    {
      return( onWarning( Warning::FILES_NOT_FOUND, &files ) );
    }

    inline  bool  PlugInCallback::onUndefinedToken( unsigned int position, const std::string &context, const std::string &token ) const
    {
      EmptyTokenInfo  uti;
      uti.position  = position;
      uti.context   = context;
      uti.token     = token;
      return( onWarning( Warning::UNDEFINED_TOKEN, &uti ) );
    }

    inline  bool  PlugInCallback::onIncompatibleValues( unsigned int position, const std::string &context, const std::string &value0Name,
                                                        int value0, const std::string &value1Name, int value1 ) const
    {
      IncompatibleValueInfo ivi;
      ivi.position    = position;
      ivi.context     = context;
      ivi.valueType   = TypeID::INT;
      ivi.value0Name  = value0Name;
      ivi.value0      = &value0;
      ivi.value1Name  = value1Name;
      ivi.value1      = &value1;
      return( onWarning( Warning::INCOMPATIBLE_VALUES, &ivi ) );
    }

    inline  bool  PlugInCallback::onIncompatibleValues( unsigned int position, const std::string &context, const std::string &value0Name,
                                                        float value0, const std::string &value1Name, float value1 ) const
    {
      IncompatibleValueInfo ivi;
      ivi.position    = position;
      ivi.context     = context;
      ivi.valueType   = TypeID::FLOAT;
      ivi.value0Name  = value0Name;
      ivi.value0      = &value0;
      ivi.value1Name  = value1Name;
      ivi.value1      = &value1;
      return( onWarning( Warning::INCOMPATIBLE_VALUES, &ivi ) );
    }


    inline  bool  PlugInCallback::onInvalidValue( unsigned int position, const std::string &context,
                                                  const std::string &valueName, int value ) const
    {
      InvalidValueInfo  ivi;
      ivi.position  = position;
      ivi.context   = context;
      ivi.valueName = valueName;
      ivi.valueType = TypeID::INT;
      ivi.value     = &value;
      return( onWarning( Warning::INVALID_VALUE, &ivi ) );
    }

    inline  bool  PlugInCallback::onInvalidValue( unsigned int position, const std::string &context,
                                                  const std::string &valueName, float value ) const
    {
      InvalidValueInfo  ivi;
      ivi.position  = position;
      ivi.context   = context;
      ivi.valueName = valueName;
      ivi.valueType = TypeID::FLOAT;
      ivi.value     = &value;
      return( onWarning( Warning::INVALID_VALUE, &ivi ) );
    }

    inline  bool  PlugInCallback::onUnsupportedToken( unsigned int position, const std::string &context, const std::string &token ) const
    {
      UnsupportedTokenInfo  uti;
      uti.position  = position;
      uti.context   = context;
      uti.token     = token;
      return( onWarning( Warning::UNSUPPORTED_TOKEN, &uti ) );
    }

    inline  void  PlugInCallback::onFileAccessFailed( const std::string& file, unsigned int systemSpecificErrorCode ) const
    {
      FileAccessFailedInfo fafi;
      fafi.file = file;
      fafi.systemSpecificErrorCode = systemSpecificErrorCode;
      onError(Error::FILE_ACCESS_FAILED, &fafi);
    }

    inline  void  PlugInCallback::onFileMappingFailed( unsigned int systemSpecificErrorCode ) const
    {
      FileMappingFailedInfo fmfi;
      fmfi.systemSpecificErrorCode = systemSpecificErrorCode;
      onError(Error::FILE_MAPPING_FAILED, &fmfi);
    }

    inline  void  PlugInCallback::onIncompatibleFile( const std::string& file, const std::string& context
                                                    , unsigned int expectedVersion, unsigned int detectedVersion) const
    {
      IncompatibleFileInfo ifi;
      ifi.file = file;
      ifi.context = context;
      ifi.expectedVersion = expectedVersion;
      ifi.detectedVersion = detectedVersion;
      onError(Error::INCOMPATIBLE_FILE, &ifi);
    }

    inline  void  PlugInCallback::onInvalidFile( const std::string& file, const std::string& context ) const
    {
      InvalidFileInfo ifi;
      ifi.file = file;
      ifi.context = context;
      onError(Error::INVALID_FILE, &ifi);
    }
  } // namespace util
} // namespace dp

