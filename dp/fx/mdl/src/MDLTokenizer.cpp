// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/DP.h>
#include <dp/Types.h>
#include <dp/Exception.h>
#include <dp/fx/mdl/inc/MDLTokenizer.h>
#include <dp/util/File.h>
#include <boost/algorithm/string.hpp>
#include <iostream>

namespace dp
{
  namespace fx
  {
    namespace mdl
    {
      static bool isHiddenMaterial( mi::base::Handle<mi::neuraylib::IAnnotation_block const> const& annotations );
      static std::string stripTrailingDigits( std::string const& name );
      static std::string typeName( mi::base::Handle<mi::neuraylib::IType const> const& type );


      static bool isHiddenMaterial( mi::base::Handle<mi::neuraylib::IAnnotation_block const> const& annotations )
      {
        if ( annotations )
        {
          for ( mi::Size i=0 ; i<annotations->get_size() ; i++ )
          {
            mi::base::Handle<mi::neuraylib::IAnnotation const> annotation = mi::base::make_handle(annotations->get_annotation(i));
            std::string annotationName = annotation->get_name();
            if ( annotationName == "::anno::hidden()" )
            {
              return( true );
            }
          }
        }
        return( false );
      }

      static std::string stripTrailingDigits( std::string const& name )
      {
        size_t pos = name.find_last_not_of( "0123456789" );
        if ( ( pos < name.length() - 1 ) && ( name[pos] == '_' ) )
        {
          return( name.substr( 0, pos ) );
        }
        return( name );
      }

      static std::string typeName( mi::base::Handle<mi::neuraylib::IType const> const& type )
      {
        mi::neuraylib::IType::Kind kind = type->get_kind();
        switch( kind )
        {
          case mi::neuraylib::IType::TK_ALIAS :
            return typeName(mi::base::make_handle(type.get_interface<mi::neuraylib::IType_alias const>()->get_aliased_type()));
          case mi::neuraylib::IType::TK_ARRAY:
            {
              mi::base::Handle<mi::neuraylib::IType_array const> array = type.get_interface<mi::neuraylib::IType_array const>();
              std::ostringstream oss;
              oss << typeName( mi::base::make_handle( array->get_element_type() ) );
              if ( array->is_immediate_sized() )
              {
                mi::Size size = array->get_size();
                oss << "[" << size << "]";
              }
              else
              {
                const char* deferredSize = array->get_deferred_size();
                DP_ASSERT(deferredSize != nullptr);
                // deferredSize can be an empty string. Normally it contains strings like "::df::L1559::N".
                // The actual size can be determined via the return type of the T[](...) function which constructs this array.
                oss << "[" << std::string( deferredSize ) << "]";
              }
              return( oss.str() );
            }
          case mi::neuraylib::IType::TK_BOOL :
            return( "Bool" );
          case mi::neuraylib::IType::TK_BSDF_MEASUREMENT :
            return( "BsdfMeasurement" );
          case mi::neuraylib::IType::TK_BSDF :
            return( "Bsdf" );
          case mi::neuraylib::IType::TK_EDF :
            return( "Edf" );
          case mi::neuraylib::IType::TK_ENUM :
            return type.get_interface<mi::neuraylib::IType_enum const>()->get_symbol();
          case mi::neuraylib::IType::TK_COLOR :
            return( "Color" );
          case mi::neuraylib::IType::TK_FLOAT :
            return( "Float" );
          case mi::neuraylib::IType::TK_INT :
            return( "Int" );
          case mi::neuraylib::IType::TK_LIGHT_PROFILE :
            return( "LightProfile" );
          case mi::neuraylib::IType::TK_MATRIX :
            {
              mi::base::Handle<mi::neuraylib::IType_matrix const> matrix = type.get_interface<mi::neuraylib::IType_matrix const>();
              DP_ASSERT(mi::base::make_handle(mi::base::make_handle(matrix->get_element_type())->get_element_type())->get_kind() == mi::neuraylib::IType::TK_FLOAT);
              std::ostringstream oss;
              oss << typeName(mi::base::make_handle(mi::base::make_handle(matrix->get_element_type())->get_element_type())) << "<" << matrix->get_size() << "," << matrix->get_size() << ">";
              return(oss.str());
            }
          case mi::neuraylib::IType::TK_STRING :
            return( "String" );
          case mi::neuraylib::IType::TK_STRUCT :
            return type.get_interface<mi::neuraylib::IType_struct const>()->get_symbol();
          case mi::neuraylib::IType::TK_TEXTURE :
            return( "Texture" );
          case mi::neuraylib::IType::TK_VDF :
            return( "Vdf" );
          case mi::neuraylib::IType::TK_VECTOR :
            {
              mi::base::Handle<mi::neuraylib::IType_vector const> vector = type.get_interface<mi::neuraylib::IType_vector const>();
              DP_ASSERT(mi::base::make_handle(vector->get_element_type())->get_kind() == mi::neuraylib::IType::TK_FLOAT);
              std::ostringstream oss;
              oss << typeName(mi::base::make_handle(vector->get_element_type())) << "<" << vector->get_size() << ">";
              return(oss.str());
            }
          default :
            DP_ASSERT( !"never passed this path!" );
            return( "" );
        }
      }


      class Logger : public mi::base::Interface_implement<mi::base::ILogger>
      {
        public:
          void message( mi::base::Message_severity level, const char* module_category, const char* message );
      };

      void Logger::message( mi::base::Message_severity level, const char* module_category, const char* message )
      {
        DP_ASSERT( strcmp( module_category, "MDL" ) == 0 );
        std::string m( message );

        std::string levelString;
        bool handled = false;
        switch ( level )
        {
          case mi::base::MESSAGE_SEVERITY_FATAL:
            levelString = "FATAL";
            break;
          case mi::base::MESSAGE_SEVERITY_ERROR:
            levelString = "ERROR";
            handled = boost::starts_with( m, "Could not locate default texture" )
                   || boost::starts_with( m, "No image plugin found to handle" );
            break;
          case mi::base::MESSAGE_SEVERITY_WARNING:
            levelString = "WARNING";
            handled = boost::contains( m, "extra parenthesis around function name" )
                   || boost::contains( m, "unused parameter" )
                   || boost::contains( m, "unused variable" )
                   || boost::ends_with( m, "material parameters of type 'bsdf' are forbidden" );
            break;
          case mi::base::MESSAGE_SEVERITY_INFO:
            levelString = "INFO";
            handled = boost::starts_with( m, "Found MDL module" )
                   || boost::starts_with( m, "Loading BSDF measurement" )
                   || boost::starts_with( m, "Loading lightprofile" );
            break;
          case mi::base::MESSAGE_SEVERITY_VERBOSE:
            levelString = "VERBOSE";
            break;
          case mi::base::MESSAGE_SEVERITY_DEBUG :
            levelString = "DEBUG";
            handled = boost::contains( m, "searching for" ) && boost::contains( m, "in path" ) && boost::contains( m, "not found" );
            break;
          default:
            levelString = "UNKNOWN";
            break;
        }
        if ( !handled )
        {
          std::cerr << "MDLTokenizer: " << levelString << ": " << message << std::endl;
          DP_ASSERT( !"MDLTokenizer Logger::message() unhandled!" );
        }
      }


      MDLTokenizer::MDLTokenizer()
        : m_filterDefaults( false )
      {
        m_mdlSDK = dp::util::DynamicLibrary::createFromFile( "libmdl_sdk.dll" );
        if ( !m_mdlSDK )
        {
          std::cerr << "MDL2XML: Can't load libmdl_sdk.dll" << std::endl;
          throw dp::FileNotFoundException( "libmdl_sdk.dll" );
        }

        typedef mi::neuraylib::INeuray* (INeuray_factory)( mi::neuraylib::IAllocator*, mi::Uint32 );
        INeuray_factory* factory = (INeuray_factory*)m_mdlSDK->getSymbol( "mi_neuray_factory" );
        DP_ASSERT( factory );

        m_neuray = factory( 0, MI_NEURAYLIB_API_VERSION );
        DP_ASSERT( m_neuray.is_valid_interface() );
        DP_VERIFY( m_neuray->start() == 0 );

        m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();

        m_mdlCompiler = m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>();
        m_mdlCompiler->set_logger( new Logger );

        mi::base::Handle<mi::neuraylib::IScope> scope(m_database->get_global_scope());
        m_transaction = mi::base::make_handle(scope->create_transaction());

        m_mdlFactory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();
        m_mdlExpressionFactory = m_mdlFactory->create_expression_factory(m_transaction.get());
        m_mdlValueFactory = m_mdlFactory->create_value_factory(m_transaction.get());

        char * mdlSystemPath = getenv( "MDL_SYSTEM_PATH" );
        if ( mdlSystemPath )
        {
          DP_VERIFY( m_mdlCompiler->add_module_path( mdlSystemPath ) == 0 );
          DP_VERIFY(m_mdlCompiler->load_module(m_transaction.get(), "::nvidia::core_definitions") == 0);
          m_transaction->commit();
          m_transaction = mi::base::make_handle(scope->create_transaction());
        }
      }

      MDLTokenizer::~MDLTokenizer()
      {
        m_database.reset();
        m_mdlExpressionFactory.reset();
        m_mdlFactory.reset();
        m_mdlCompiler.reset();    // throw away before m_neuray is shut down
        m_neuray->shutdown();
      }

      bool MDLTokenizer::checkDefaultField(std::string const& fieldName, mi::base::Handle<mi::neuraylib::IExpression const> const& expression)
      {
        bool isDefault = (expression->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT);
        if (isDefault)
        {
          if (fieldName == "ior")
          {
            isDefault = (m_mdlExpressionFactory->compare(expression.get(), m_mdlExpressionFactory->create_constant(m_mdlValueFactory->create_color(1.0f, 1.0f, 1.0f))) == 0);
          }
          else if (fieldName == "thin_walled")
          {
            isDefault = (m_mdlExpressionFactory->compare(expression.get(), m_mdlExpressionFactory->create_constant(m_mdlValueFactory->create_bool(false))) == 0);
          }
        }
        return(isDefault);
      }

      void MDLTokenizer::parseFile(std::string const& file, dp::util::FileFinder const& fileFinder)
      {
        DP_ASSERT( dp::util::fileExists( file ) );
        if (fileBegin(file))
        {
          std::vector<std::string> searchPaths = fileFinder.getSearchPaths();
          std::string packagePath = dp::util::getFilePath(file);
          if (!searchPaths.empty() && boost::starts_with(file, searchPaths.front()))
          {
            packagePath = searchPaths.front();
          }
          m_mdlCompiler->add_module_path(packagePath.c_str());

          for (auto const& s : searchPaths)
          {
            m_mdlCompiler->add_module_path(s.c_str()); // add_resource_path() is deprecated.
          }

          std::string moduleName = dp::util::getFilePath(file) + "\\" + dp::util::getFileStem(file);
          DP_ASSERT(boost::starts_with(moduleName, packagePath));
          boost::erase_first(moduleName, packagePath);
          boost::replace_all(moduleName, "\\", "::");
          boost::replace_all(moduleName, "/", "::");

          mi::Sint32 reason = m_mdlCompiler->load_module(m_transaction.get(), moduleName.c_str());
          if (reason < 0)
          {
            std::cout << "MDLTokenizer: failed to load module <" << file << ">" << std::endl;
            switch (reason)
            {
              // case 1: // Success (module exists already, loading from file was skipped).
              // case 0: // Success (module was actually loaded from file).
              case -1:
                std::cout << "The module name is invalid or a NULL pointer." << std::endl;
                break;
              case -2:
                std::cout << "Failed to find or to compile the module." << std::endl;
                break;
              case -3:
                std::cout << "The database name for an imported module is already in use but is not an MDL module," << std::endl;
                std::cout << "or the database name for a definition in this module is already in use." << std::endl;
                break;
              case -4:
                std::cout << "Initialization of an imported module failed." << std::endl;
                break;
              default:
                std::cout << "Unexpected return value " << reason << " from IMdl_compiler::load_module()." << std::endl;
                DP_ASSERT(!"Unexpected return value from IMdl_compiler::load_module()");
                break;
            }
          }
          else
          {
            moduleName = std::string("mdl") + moduleName;
            mi::base::Handle<const mi::neuraylib::IModule> module = mi::base::make_handle(m_transaction->access<mi::neuraylib::IModule>(moduleName.c_str()));
            DP_ASSERT(module.is_valid_interface());
            DP_ASSERT(module->get_filename());
            std::string fn = module->get_filename();

            mi::Size materialCount = module->get_material_count();
            DP_ASSERT(0 <= materialCount);
            for (mi::Size i = 0; i < materialCount; i++)
            {
              m_materialDefinition = mi::base::make_handle(m_transaction->access<mi::neuraylib::IMaterial_definition>(module->get_material(i)));
              DP_ASSERT(m_materialDefinition);

              if (!isHiddenMaterial(mi::base::make_handle(m_materialDefinition->get_annotations())))
              {
                mi::Sint32 result;
                mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance = mi::base::make_handle(m_materialDefinition->create_material_instance(0, &result));
                if (result == 0)
                {
                  m_compiledMaterial = mi::base::make_handle(materialInstance->create_compiled_material(mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, 1.0f, &result));
                  DP_ASSERT(result == 0);

                  tokenizeMaterial(module->get_material(i));
                  m_compiledMaterial.reset();
                }
                else
                {
                  std::cout << "failed to create a material instance for material <" << m_materialDefinition->get_mdl_name() << ">" << std::endl;
                  switch (result)
                  {
                    case -1:
                      std::cout << "An argument for a non-existing parameter was provided in arguments." << std::endl;
                      break;
                    case -2:
                      std::cout << "The type of an argument in arguments does not have the correct type, see get_parameter_types()." << std::endl;
                      break;
                    case -3:
                      std::cout << "A parameter that has no default was not provided with an argument value." << std::endl;
                      break;
                    case -4:
                      std::cout << "The definition can not be instantiated because it is not exported." << std::endl;
                      break;
                    case -5:
                      std::cout << "A parameter type is uniform, but the corresponding argument has a varying return type." << std::endl;
                      break;
                    case -6:
                      std::cout << "An argument expression is not a constant nor a call." << std::endl;
                      break;
                    default:
                      std::cout << "ERROR: Unexpected error result from create_material_instance()" << std::endl;
                      break;
                  }
                }
              }
              m_materialDefinition.reset();
            }
          }

          for (auto const& s : searchPaths)
          {
            m_mdlCompiler->remove_resource_path(s.c_str());
          }
          m_mdlCompiler->remove_module_path(packagePath.c_str());

          fileEnd();
        }
      }

      void MDLTokenizer::setFilterDefaults(bool filter)
      {
        m_filterDefaults = filter;
      }

      void MDLTokenizer::triggerTokenizeFunctionReturnType(std::string functionName)
      {
        if ( boost::starts_with( functionName, "mdl::base" ) )
        {
          mi::base::Handle<const mi::neuraylib::IModule> module = mi::base::make_handle(m_transaction->access<mi::neuraylib::IModule>("mdl::base"));
          mi::base::Handle<const mi::IArray> overloads = mi::base::make_handle(module->get_function_overloads(functionName.c_str()));
          DP_ASSERT(overloads->get_length() == 1);
          mi::base::Handle<const mi::IString> mdlFunctionName = mi::base::make_handle(overloads->get_element<mi::IString>(static_cast<mi::Uint32>(0)));
          mi::base::Handle<const mi::neuraylib::IFunction_definition> functionDefinition = mi::base::make_handle(m_transaction->access<mi::neuraylib::IFunction_definition>(mdlFunctionName->get_c_str()));
          tokenizeType(mi::base::make_handle(functionDefinition->get_return_type()));
        }
      }

      void MDLTokenizer::tokenizeArgument(mi::Size idx, std::string const& name, mi::base::Handle<mi::neuraylib::IExpression const> const& argumentExpression, mi::base::Handle<mi::neuraylib::IExpression const> const& defaultExpression)
      {
        if (!(m_filterDefaults && defaultExpression && (m_mdlExpressionFactory->compare(argumentExpression.get(), defaultExpression.get()) == 0)))
        {
          if (argumentBegin(dp::checked_cast<unsigned int>(idx), typeName(mi::base::make_handle(argumentExpression->get_type())), name))
          {
            tokenizeExpression(argumentExpression);
            argumentEnd();
          }
        }
      }

      void MDLTokenizer::tokenizeArray(mi::base::Handle<mi::neuraylib::IValue_array const> const& value)
      {
        mi::base::Handle<mi::neuraylib::IType const> type = mi::base::make_handle(value->get_type());
        tokenizeType(type);

        if (arrayBegin(typeName(type), value->get_size()))
        {
          for (mi::Size i = 0; i < value->get_size(); i++)
          {
            tokenizeValue(mi::base::make_handle(value->get_value(i)));
          }
          arrayEnd();
        }
      }

      void MDLTokenizer::tokenizeBSDFMeasurement(mi::base::Handle<mi::neuraylib::IValue_bsdf_measurement const> const& value)
      {
        if ( value )
        {
          DP_ASSERT( value->get_value() );
          std::string fileName = value->get_value();
          DP_ASSERT( boost::starts_with( fileName, "MI_default_bsdf_measurement_" ) );
          boost::erase_head( fileName, static_cast<int>( strlen( "MI_default_bsdf_measurement_" ) ) );
          valueBsdfMeasurement( fileName );
        }
        else
        {
          defaultRef( "BsdfMeasurement" );
        }
      }

      void MDLTokenizer::tokenizeColor(mi::base::Handle<mi::neuraylib::IValue_color const> const& value)
      {
        DP_ASSERT( value->get_size() == 3 );
        dp::math::Vec3f colorValue;
        for ( mi::Size i=0 ; i<3 ; i++ )
        {
          colorValue[i] = mi::base::make_handle(value->get_value(i))->get_value();
        }
        valueColor( colorValue );
      }

      void MDLTokenizer::tokenizeDirectCall(mi::base::Handle<mi::neuraylib::IExpression_direct_call const> const& call)
      {
        // special handling for functions containing the substring "_legacy" -> remove that part
        std::string callDefinition = call->get_definition();
        size_t pos = callDefinition.find("_legacy");
        if (pos != std::string::npos)
        {
          callDefinition.erase(pos, strlen("_legacy"));
        }

        mi::base::Handle<mi::neuraylib::IType const> type = mi::base::make_handle(call->get_type());
        tokenizeType(type);
        if (callBegin(typeName(type), callDefinition))
        {
          mi::base::Handle<mi::neuraylib::IFunction_definition const> functionDefinition = mi::base::make_handle(m_transaction->access<mi::neuraylib::IFunction_definition>(callDefinition.c_str()));
          mi::base::Handle<mi::neuraylib::IExpression_list const> defaults = mi::base::make_handle(functionDefinition->get_defaults());

          mi::base::Handle<mi::neuraylib::IExpression_list const> arguments = mi::base::make_handle(call->get_arguments());
          for (mi::Size i = 0; i<arguments->get_size(); i++)
          {
            tokenizeArgument(i, arguments->get_name(i), mi::base::make_handle(arguments->get_expression(i)),
                             mi::base::make_handle(defaults->get_expression(functionDefinition->get_parameter_name(i))));
          }
          callEnd();
        }
      }

      void MDLTokenizer::tokenizeEnum( mi::base::Handle<mi::neuraylib::IValue_enum const> const& value )
      {
        mi::base::Handle<mi::neuraylib::IType_enum const> type = mi::base::make_handle(value->get_type());
        tokenizeEnumType(type);
        valueEnum( type->get_symbol(), value->get_value(), type->get_value_name( value->get_index() ) );
      }

      void MDLTokenizer::tokenizeEnumType( mi::base::Handle<mi::neuraylib::IType_enum const> const& type )
      {
        if (enumTypeBegin(type->get_symbol(), type->get_size()))
        {
          for ( mi::Size i=0 ; i<type->get_size() ; i++ )
          {
            enumTypeValue( type->get_value_name( i ), type->get_value_code( i ) );
          }
          enumTypeEnd();
        }
      }

      void MDLTokenizer::tokenizeExpression( mi::base::Handle<mi::neuraylib::IExpression const> const& expression )
      {
        switch( expression->get_kind() )
        {
          case mi::neuraylib::IExpression::EK_CALL :
            DP_ASSERT( !"never passed this path!" );
            break;
          case mi::neuraylib::IExpression::EK_CONSTANT :
            tokenizeValue(mi::base::make_handle(expression.get_interface<mi::neuraylib::IExpression_constant const>()->get_value()));
            break;
          case mi::neuraylib::IExpression::EK_DIRECT_CALL :
            tokenizeDirectCall(expression.get_interface<mi::neuraylib::IExpression_direct_call const>());
            break;
          case mi::neuraylib::IExpression::EK_PARAMETER :
            tokenizeParameterExpression(expression.get_interface<mi::neuraylib::IExpression_parameter const>());
            break;
          case mi::neuraylib::IExpression::EK_TEMPORARY :
            referenceTemporary(dp::checked_cast<unsigned int>(expression.get_interface<mi::neuraylib::IExpression_temporary const>()->get_index()));
            break;
          default :
            DP_ASSERT( false );
            break;
        }
      }

      void MDLTokenizer::tokenizeField( std::string const& fieldName )
      {
        mi::base::Handle<mi::neuraylib::IExpression const> expression = mi::base::make_handle(m_compiledMaterial->get_field(fieldName.c_str()));
        if (!(m_filterDefaults && checkDefaultField(fieldName, expression)))
        {
          if ( fieldBegin( fieldName ) )
          {
            tokenizeExpression( expression );
            fieldEnd();
          }
        }
      }

      void MDLTokenizer::tokenizeLightProfile(mi::base::Handle<mi::neuraylib::IValue_light_profile const> const& value)
      {
        if ( value )
        {
          DP_ASSERT( value->get_value() );
          std::string fileName = value->get_value();
          DP_ASSERT( boost::starts_with( fileName, "MI_default_lightprofile_" ) );
          boost::erase_head( fileName, static_cast<int>( strlen( "MI_default_lightprofile_" ) ) );
          valueLightProfile( fileName );
        }
        else
        {
          defaultRef( "LightProfile" );
        }
      }

      void MDLTokenizer::tokenizeMaterial( std::string const& name )
      {
        mi::base::Uuid hash = m_compiledMaterial->get_hash();

        if (materialBegin(name, dp::math::Vec4ui(hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4)))
        {
          mi::Size parameterCount = m_compiledMaterial->get_parameter_count();
          for (mi::Size i = 0; i < parameterCount; i++)
          {
            tokenizeParameter(i);
          }

          mi::Size temporaryCount = m_compiledMaterial->get_temporary_count();
          for (mi::Uint32 ti = 0; ti < temporaryCount; ti++)
          {
            tokenizeTemporary(ti);
          }

          tokenizeField("thin_walled");
          tokenizeField("surface");
          tokenizeField("backface");
          tokenizeField("ior");
          tokenizeField("volume");
          tokenizeField("geometry");

          materialEnd();
        }
      }

      void MDLTokenizer::tokenizeMatrix( mi::base::Handle<mi::neuraylib::IValue_matrix const> const& value )
      {
        if (matrixBegin(typeName(mi::base::make_handle(value->get_type()))))
        {
          for (mi::Size i = 0; i < value->get_size(); i++)
          {
            tokenizeVector(mi::base::make_handle(value->get_value(i)));
          }
          matrixEnd();
        }
      }

      void MDLTokenizer::tokenizeParameter( mi::Size parameterIndex )
      {
        char const* parameterName = m_compiledMaterial->get_parameter_name( parameterIndex );
        if (parameterBegin(dp::checked_cast<unsigned int>(parameterIndex), parameterName ? parameterName : ""))
        {
          tokenizeValue(mi::base::make_handle(m_compiledMaterial->get_argument(parameterIndex)));

          // only named parameters need annotations!!
          if (parameterName)
          {
            mi::base::Handle<mi::neuraylib::IAnnotation_list const> annotations = mi::base::make_handle(m_materialDefinition->get_parameter_annotations());
            DP_ASSERT(annotations);

            mi::base::Handle<mi::neuraylib::IAnnotation_block const> parameterAnnotations = mi::base::make_handle(annotations->get_annotation_block(parameterName));
            if (parameterAnnotations)
            {
              for (mi::Size i = 0; i < parameterAnnotations->get_size(); i++)
              {
                mi::base::Handle<mi::neuraylib::IAnnotation const> annotation = mi::base::make_handle(parameterAnnotations->get_annotation(i));
                std::string annotationName = annotation->get_name();
                if (annotationBegin(annotationName))
                {
                  mi::base::Handle<mi::neuraylib::IExpression_list const> annotationArguments = mi::base::make_handle(annotation->get_arguments());
                  for (mi::Size j = 0; j < annotationArguments->get_size(); j++)
                  {
                    tokenizeExpression(mi::base::make_handle(annotationArguments->get_expression(j)));
                  }
                  annotationEnd();
                }
              }
            }
          }

          parameterEnd();
        }
      }

      void MDLTokenizer::tokenizeParameterExpression( mi::base::Handle<mi::neuraylib::IExpression_parameter const> const& expression )
      {
        tokenizeType(mi::base::make_handle(expression->get_type()));
        referenceParameter(dp::checked_cast<unsigned int>(expression->get_index()));
      }

      void MDLTokenizer::tokenizeStructure( mi::base::Handle<mi::neuraylib::IValue_struct const> const& structValue )
      {
        mi::base::Handle<mi::neuraylib::IType_struct const> type = mi::base::make_handle(structValue->get_type());
        tokenizeStructureType(type);

        if (structureBegin(type->get_symbol()))
        {
          for (mi::Size i = 0; i < structValue->get_size(); i++)
          {
            tokenizeValue(mi::base::make_handle(structValue->get_value(i)));
          }
          structureEnd();
        }
      }

      void MDLTokenizer::tokenizeStructureType( mi::base::Handle<mi::neuraylib::IType_struct const> const& type )
      {
        if (structureTypeBegin( type->get_symbol() ))
        {
          for ( mi::Size i=0 ; i<type->get_size() ; i++ )
          {
            mi::base::Handle<mi::neuraylib::IType const> componentType = mi::base::make_handle(type->get_component_type(i));
            tokenizeType(componentType);
            structureTypeElement( typeName( componentType ), type->get_field_name( i ) );
          }
          structureTypeEnd();
        }
      }

      void MDLTokenizer::tokenizeTemporary( mi::Size temporaryIndex )
      {
        if (temporaryBegin(dp::checked_cast<unsigned int>(temporaryIndex)))
        {
          tokenizeExpression(mi::base::make_handle(m_compiledMaterial->get_temporary(temporaryIndex)));
          temporaryEnd();
        }
      }

      void MDLTokenizer::tokenizeTexture( mi::base::Handle<mi::neuraylib::IValue_texture const> const& value )
      {
        if ( value )
        {
          DP_ASSERT( value->get_value() );
          mi::base::Handle<mi::neuraylib::ITexture const> texture = mi::base::make_handle(m_transaction->access<mi::neuraylib::ITexture>(value->get_value()));
          std::string fileName = texture->get_image();
          DP_ASSERT( boost::starts_with( fileName, "MI_default_image_" ) );
          boost::erase_head( fileName, static_cast<int>( strlen( "MI_default_image_" ) ) );
          mi::Float32 gamma = texture->get_effective_gamma();
          DP_ASSERT( ( gamma == 2.2f ) || ( gamma == 1.0f ) || ( gamma == 0.0f ) );   // I know, operator==() on floating points is unsafe! But here it works!
          GammaMode gammaMode = ( gamma == 2.2f ) ? GammaMode::SRGB : ( gamma == 1.0f ) ? GammaMode::LINEAR : GammaMode::DEFAULT;
          valueTexture( fileName, gammaMode );
        }
        else
        {
          defaultRef( "Texture" );
        }
      }

      void MDLTokenizer::tokenizeType( mi::base::Handle<mi::neuraylib::IType const> const& type )
      {
        mi::neuraylib::IType::Kind kind = type->get_kind();
        switch( kind )
        {
          case mi::neuraylib::IType::TK_ALIAS :
            tokenizeType(mi::base::make_handle(type.get_interface<mi::neuraylib::IType_alias const>()->get_aliased_type()));
            break;
          case mi::neuraylib::IType::TK_ARRAY :
            tokenizeType(mi::base::make_handle(type.get_interface<mi::neuraylib::IType_array const>()->get_element_type()));
            break;
          case mi::neuraylib::IType::TK_ENUM :
            tokenizeEnumType(type.get_interface<mi::neuraylib::IType_enum const>());
            break;
          case mi::neuraylib::IType::TK_MATRIX :
            tokenizeType(mi::base::make_handle(type.get_interface<mi::neuraylib::IType_matrix const>()->get_element_type()));
            break;
          case mi::neuraylib::IType::TK_STRUCT :
            tokenizeStructureType(type.get_interface<mi::neuraylib::IType_struct const>());
            break;
          case mi::neuraylib::IType::TK_VECTOR :
            tokenizeType(mi::base::make_handle(type.get_interface<mi::neuraylib::IType_vector const>()->get_element_type()));
            break;
          case mi::neuraylib::IType::TK_BOOL :   // those don't need explicit parsing!
          case mi::neuraylib::IType::TK_BSDF :
          case mi::neuraylib::IType::TK_BSDF_MEASUREMENT :
          case mi::neuraylib::IType::TK_COLOR :
          case mi::neuraylib::IType::TK_EDF :
          case mi::neuraylib::IType::TK_FLOAT :
          case mi::neuraylib::IType::TK_INT :
          case mi::neuraylib::IType::TK_LIGHT_PROFILE :
          case mi::neuraylib::IType::TK_TEXTURE :
          case mi::neuraylib::IType::TK_VDF :
            break;
          default :
            DP_ASSERT( !"never passed this path!" );
            break;
        }
      }

      void MDLTokenizer::tokenizeValue( mi::base::Handle<mi::neuraylib::IValue const> const& value )
      {
        mi::base::Handle<mi::neuraylib::IType const> type = mi::base::make_handle(value->get_type());

        mi::neuraylib::IValue::Kind kind = value->get_kind();
        switch( kind )
        {
          case mi::neuraylib::IValue::VK_ARRAY :
            tokenizeArray(value.get_interface<mi::neuraylib::IValue_array const>());
            break;
          case mi::neuraylib::IValue::VK_BOOL :
            valueBool(value.get_interface<mi::neuraylib::IValue_bool const>()->get_value());
            break;
          case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT :
            tokenizeBSDFMeasurement(value.get_interface<mi::neuraylib::IValue_bsdf_measurement const>());
            break;
          case mi::neuraylib::IValue::VK_COLOR :
            tokenizeColor(value.get_interface<mi::neuraylib::IValue_color const>());
            break;
          case mi::neuraylib::IValue::VK_ENUM :
            tokenizeEnum(value.get_interface<mi::neuraylib::IValue_enum const>());
            break;
          case mi::neuraylib::IValue::VK_FLOAT :
            valueFloat(value.get_interface<mi::neuraylib::IValue_float const>()->get_value());
            break;
          case mi::neuraylib::IValue::VK_INT :
            valueInt(value.get_interface<mi::neuraylib::IValue_int const>()->get_value());
            break;
          case mi::neuraylib::IValue::VK_INVALID_REF :
            switch( type->get_kind() )
            {
              case mi::neuraylib::IType::TK_BSDF :
                defaultRef( "Bsdf" );
                break;
              case mi::neuraylib::IType::TK_EDF :
                defaultRef( "Edf" );
                break;
              case mi::neuraylib::IType::TK_LIGHT_PROFILE :
                defaultRef( "LightProfile" );
                break;
              case mi::neuraylib::IType::TK_TEXTURE :
                defaultRef( "Texture" );
                break;
              case mi::neuraylib::IType::TK_VDF :
                defaultRef( "Vdf" );
                break;
              default :
                DP_ASSERT( false );
                break;
            }
            break;
          case mi::neuraylib::IValue::VK_LIGHT_PROFILE :
            tokenizeLightProfile(value.get_interface<mi::neuraylib::IValue_light_profile const>());
            break;
          case mi::neuraylib::IValue::VK_MATRIX :
            tokenizeMatrix(value.get_interface<mi::neuraylib::IValue_matrix const>());
            break;
          case mi::neuraylib::IValue::VK_STRING :
            valueString(value.get_interface<mi::neuraylib::IValue_string const>()->get_value());
            break;
          case mi::neuraylib::IValue::VK_STRUCT :
            tokenizeStructure(value.get_interface<mi::neuraylib::IValue_struct const>());
            break;
          case mi::neuraylib::IValue::VK_TEXTURE :
            tokenizeTexture(value.get_interface<mi::neuraylib::IValue_texture const>());
            break;
          case mi::neuraylib::IValue::VK_VECTOR :
            tokenizeVector(value.get_interface<mi::neuraylib::IValue_vector const>());
            break;
          default :
            DP_ASSERT( !"never passed this path!" );
            break;
        }
      }

      void MDLTokenizer::tokenizeVector( mi::base::Handle<mi::neuraylib::IValue_vector const> const& value )
      {
        if (vectorBegin(typeName(mi::base::make_handle(value->get_type()))))
        {
          for (mi::Size i = 0; i < value->get_size(); i++)
          {
            tokenizeValue(mi::base::make_handle(value->get_value(i)));
          }
          vectorEnd();
        }
      }

    } // mdl
  } // fx
} // dp
