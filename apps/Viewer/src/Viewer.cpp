// Copyright NVIDIA Corporation 2009-2015
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


#include "Log.h"
#include "Preferences.h"
#include "Viewer.h"
#include "viewerPlugInCallback.h"

#include <QPixmap>

#include <boost/assign/list_of.hpp>
#include <boost/program_options.hpp>

#include <dp/fx/EffectLibrary.h>
#include <dp/sg/algorithm/IndexTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/io/IO.h>


// perhaps this method should be in ScriptSystem?
static QScriptValue LogMethod(QScriptContext *context, QScriptEngine *engine)
{
  QString result;
  for (int i = 0; i < context->argumentCount(); ++i) 
  {
    if (i > 0)
    {
      result.append(" ");
    }

    result.append(context->argument(i).toString());
  }

  LogMessage( result.toStdString().c_str() );

  return engine->undefinedValue();
}

dp::culling::Mode determineCullingMode( std::string const & name )
{
  static std::map<std::string, dp::culling::Mode> cullingEngines = boost::assign::map_list_of
    ( "auto",       dp::culling::MODE_AUTO )
    ( "cpu",        dp::culling::MODE_CPU )
    ( "cuda",       dp::culling::MODE_CUDA )
    ( "gl_compute", dp::culling::MODE_OPENGL_COMPUTE );

  std::map<std::string, dp::culling::Mode>::const_iterator it = cullingEngines.find( name );
  DP_ASSERT( it != cullingEngines.end() );
  return( it->second );
}

dp::fx::Manager determineShaderManagerType( std::string const& name )
{
  static std::map<std::string, dp::fx::Manager> shaderManager = boost::assign::map_list_of
    ( "rixfx:uniform",          dp::fx::MANAGER_UNIFORM )
    ( "rixfx:shaderbufferload", dp::fx::MANAGER_SHADERBUFFER )
    ( "rix:ubo140",             dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX )
    ( "rixfx:ubo140",           dp::fx::MANAGER_UNIFORM_BUFFER_OBJECT_RIX_FX )
    ( "rixfx:ssbo140",          dp::fx::MANAGER_SHADER_STORAGE_BUFFER_OBJECT );

  std::map<std::string, dp::fx::Manager>::const_iterator it = shaderManager.find( name );
  DP_ASSERT( it != shaderManager.end() );
  return( it->second );
}

dp::sg::renderer::rix::gl::TransparencyMode determineTransparencyMode( std::string const & name )
{
  static std::map<std::string,dp::sg::renderer::rix::gl::TransparencyMode> transparencyModes = boost::assign::map_list_of
    ( "None", dp::sg::renderer::rix::gl::TM_NONE )
    ( "OITAll", dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_ALL )
    ( "OITClosestList",  dp::sg::renderer::rix::gl::TM_ORDER_INDEPENDENT_CLOSEST_LIST )
    ( "SB", dp::sg::renderer::rix::gl::TM_SORTED_BLENDED );

  std::map<std::string,dp::sg::renderer::rix::gl::TransparencyMode>::const_iterator it = transparencyModes.find( name );
  DP_ASSERT( it != transparencyModes.end() );
  return( it->second );
}

void Viewer::parseCommandLine( int & argc, char ** argv )
{
  boost::program_options::options_description od("Usage: Viewer");
  od.add_options()
    ( "backdrop", boost::program_options::value<bool>()->default_value(true), "true|false" )
    ( "cullingengine", boost::program_options::value<std::string>()->default_value("auto"), "auto|cpu|cuda|gl_compute")
    ( "file", boost::program_options::value<std::string>()->default_value(""), "file to load" )
    ( "height", boost::program_options::value<int>()->default_value(0), "Application height" )
    ( "renderengine", boost::program_options::value<std::string>()->default_value("Bindless"), "choose a renderengine from this list: VBO|VAB|VBOVAO|Bindless|BindlessVAO|DisplayList" )
    ( "script", boost::program_options::value<std::string>()->default_value(""), "script to run" )
    ( "shadermanager", boost::program_options::value<std::string>()->default_value("rix:ubo140"), "rixfx:uniform|rixfx:ubo140|rixfx:ssbo140|rixfx:shaderbufferload" )
    ( "tonemapper", boost::program_options::value<bool>()->default_value(false), "true|false" )
    ( "transparency", boost::program_options::value<std::string>()->default_value("OITAll"), "None|OITAll|OITClosestList|SB" )
    ( "width", boost::program_options::value<int>()->default_value(0), "Application width" )
    ;

  try
  {
    boost::program_options::variables_map opts;
    boost::program_options::store( boost::program_options::parse_command_line( argc, argv, od ), opts );
    boost::program_options::notify( opts );
    if ( !opts["help"].empty() )
    {
      std::cout << od << std::endl;
    }
     
    m_tonemapperEnabled = opts["tonemapper"].as<bool>();
    m_backdropEnabled = opts["backdrop"].as<bool>();  // Force to false when --raytracing true to not do it twice since the miss program renders the environment anyway.
    m_cullingMode = determineCullingMode( opts["cullingengine"].as<std::string>() );
    m_startupFile = opts["file"].as<std::string>().c_str();
    m_height = opts["height"].as<int>();
    m_renderEngine = opts["renderengine"].as<std::string>();
    m_runScript = opts["script"].as<std::string>().c_str();
    m_shaderManagerType = determineShaderManagerType( opts["shadermanager"].as<std::string>() );
    if ( !opts["transparency"].empty() )
    {
      m_preferences->setTransparencyMode( determineTransparencyMode( opts["transparency"].as<std::string>() ) );
    }
    m_width = opts["width"].as<int>();
  }
  catch ( boost::program_options::unknown_option e )
  {
    std::cerr << "Unknown option: " << e.get_option_name() << ". ";
    std::cout << od << std::endl;
  }
}

Viewer::Viewer( int & argc, char ** argv )
: QApplication( argc, argv )
, m_mainWindow(0)
, m_scriptSystem(0)
, m_displayedSceneName()
, m_preferences(0)
, m_globalShareGLWidget(0)
, m_scriptTimer(this)
, m_parameterUndoStack(this)
, m_sceneStateUndoStack(this)
, m_width(0)
, m_height(0)
, m_cullingMode(dp::culling::MODE_AUTO)
, m_renderEngine("Bindless")
{
  processEvents();

  m_preferences = new Preferences( this );

  // parse command line arguments before any further init
  parseCommandLine( argc, argv );

  // Create a global GL widget for resource sharing.
  // Needs to be done in the QApplication and not in the MainWindow
  // because the destruction of the MainWindow happens before all GL
  // resources are cleaned up.
  m_globalShareGLWidget = new dp::sg::ui::qt5::SceniXQGLWidget(0, dp::gl::RenderContextFormat() );

  // add script system
  // DAR FIXME: The ScriptSystem generates memory leak reports on program exit!
  m_scriptSystem = new ScriptSystem();
  m_scriptSystem->addFunction( "log", LogMethod ); 
  // print is more common in javascript - but assign to the same function
  m_scriptSystem->addFunction( "print", LogMethod ); 

  connect( m_preferences, SIGNAL(environmentEnabledChanged()), this, SLOT(setEnvironmentEnabledChanged()) );
  connect( m_preferences, SIGNAL(environmentTextureNameChanged(const QString&)), this, SLOT(setEnvironmentTextureName(const QString&)) );

  dp::sg::core::TextureFileSharedPtr textureFile = dp::sg::core::TextureFile::create( m_preferences->getEnvironmentTextureName().toStdString(), dp::sg::core::TT_TEXTURE_2D );
  textureFile->incrementMipmapUseCount();

  m_environmentSampler = dp::sg::core::Sampler::create( textureFile );
  m_environmentSampler->setMagFilterMode( dp::sg::core::TFM_MAG_LINEAR );
  m_environmentSampler->setMinFilterMode( dp::sg::core::TFM_MIN_LINEAR_MIPMAP_LINEAR );
  m_environmentSampler->setWrapModes( dp::sg::core::TWM_REPEAT, dp::sg::core::TWM_CLAMP_TO_EDGE, dp::sg::core::TWM_CLAMP_TO_EDGE );
}

void Viewer::setEnvironmentEnabledChanged()
{
  emit viewChanged();
}

void Viewer::setEnvironmentTextureName( const QString & name )
{
  dp::sg::core::TextureFileSharedPtr textureFile = dp::sg::core::TextureFile::create( name.toStdString(), dp::sg::core::TT_TEXTURE_2D );
  textureFile->incrementMipmapUseCount();

  m_environmentSampler->setTexture( textureFile );

  emit environmentChanged(); // update the backdrop renderer
  emit viewChanged();
}

void Viewer::setTonemapperEnabled( bool enabled )
{
  if ( m_tonemapperEnabled != enabled )
  {
    m_mainWindow->getCurrentViewport()->setTonemapperEnabled( enabled );
    m_tonemapperEnabled = enabled;
  }
}

// if successful, will set displayedScene and displayedSceneName
bool Viewer::loadScene( const QString & fileName )
{
  if ( ! fileName.isEmpty() && ( fileName != m_displayedSceneName ) )
  {
    if ( !m_viewerPlugInCallback )
    {
      m_viewerPlugInCallback = viewerPlugInCallback::create();
    }
    m_viewState.reset();    // clear the current scene before loading the next one!
    try
    {
      m_viewState = dp::sg::io::loadScene( fileName.toStdString(), GetPreferences()->getSearchPathsAsStdVector(), m_viewerPlugInCallback );
      if ( m_viewState )
      {
        LogMessage( "Loading Scene: '%s' - SUCCESS\n", fileName.toStdString().c_str() );
        m_displayedSceneName = fileName;
      }
      else
      {
        LogError( "Loading Scene: '%s' - FAILED\n", fileName.toStdString().c_str() );
        m_displayedSceneName.clear();
      }
    }
    catch ( std::exception const & e )
    {
      LogError( e.what() );
      m_displayedSceneName.clear();
    }
    emit sceneChanged();
    m_sceneStateUndoStack.clear();
    m_parameterUndoStack.clear();
    outputStatistics();
  }
  return( !!m_viewState );
}

void Viewer::unloadScene()
{
  if ( m_viewState )
  {
    m_viewState.reset();
    m_displayedSceneName.clear();
    emit sceneChanged();
    m_sceneStateUndoStack.clear();
    m_parameterUndoStack.clear();
    outputStatistics();
  }
}

bool Viewer::saveScene( const QString & fileName ) const
{
  if ( !fileName.isEmpty() )
  {
    // Saving a scene requires a ViewState with a scene and a camera.
    // DAR FIXME Better would be to use the ViewState of the currently active viewport here.
    DP_ASSERT( m_viewState && m_viewState->getScene() );
    dp::sg::core::CameraSharedPtr viewStateCamera = m_viewState->getCamera();
    if ( ! viewStateCamera )
    {
      dp::sg::core::PerspectiveCameraSharedPtr perspectiveCamera( dp::sg::core::PerspectiveCamera::create() );
      {
        perspectiveCamera->setName("ViewCamera");

        // When creating a new camera, make the scene fit into the viewport.
        if ( m_viewState->getScene()->getRootNode() )
        {
          const dp::math::Sphere3f & bs = m_viewState->getScene()->getRootNode()->getBoundingSphere();
          perspectiveCamera->zoom(bs, dp::math::PI_QUARTER);
        }

        m_viewState->setCamera( perspectiveCamera );
      }
    }

    return dp::sg::io::saveScene( fileName.toStdString(), m_viewState );
  }

  return false;
}

void Viewer::runStartupFile()
{
  m_mainWindow->loadFile( m_startupFile, false );
}

void
Viewer::runStartupScript()
{
  m_scriptSystem->executeFile( m_runScript, "commandLineScript" );
}

void
Viewer::startup()
{
  LogMessage("Welcome to Viewer.\n");

  // create and display main window
  m_mainWindow = new MainWindow();
  if ( m_width || m_height )
  {
    m_mainWindow->resize( m_width, m_height );
  }
  m_mainWindow->show();

  // need to do this after main window has been created
  QScriptValue topLevel = m_scriptSystem->addObject( VIEWER_APPLICATION_NAME, this );
  m_scriptSystem->addSubObject( topLevel, "mainWindow", m_mainWindow ); 
  m_scriptSystem->addSubObject( topLevel, "preferences", m_preferences ); 
  m_scriptSystem->addSubObject( topLevel, "timer", &m_scriptTimer ); 

  // run any startup script, if they have specified one
  if( !m_runScript.isEmpty() )
  {
    // must run startup script after app::exec() has been called, or some things don't
    // work as expected..
    QTimer::singleShot( 10, this, SLOT(runStartupScript()) );
  }
  else if ( !m_startupFile.isEmpty() )
  {
    QTimer::singleShot( 10, this, SLOT(runStartupFile()) );
  }
}

Viewer::~Viewer()
{
  DP_ASSERT( m_globalShareGLWidget );

  delete m_mainWindow;
  m_mainWindow = 0;

  delete m_scriptSystem;
  m_scriptSystem = 0;
  
  // note that this should not be executed before deleting the main window since latter operation does change the active GL context.
  if ( m_globalShareGLWidget->getRenderContext() ) // When pixelformat selection failed there is no context.
  {
    m_globalShareGLWidget->getRenderContext()->makeCurrent();
  }

  // clear stacks before sharewidget too
  m_parameterUndoStack.clear();
  m_sceneStateUndoStack.clear();

  delete m_globalShareGLWidget;
  m_globalShareGLWidget = 0;
  // preferences will be deleted automatically..
}

void
Viewer::log( const char * format, va_list valist, LogWidget::Severity severity ) const
{
  static QString preOpenBuffer;

  QString message;
  message.vsprintf( format, valist );

  // No logging posible until MainWindow creation is finished!
  // Hit while loading the MaterialEditor preview scenes during creation.
  if (m_mainWindow)
  {
    LogWidget * lw = static_cast< LogWidget * >( m_mainWindow->getLog() );

    if( !preOpenBuffer.isEmpty() )
    {
      // we don't remember the severity of these messages
      lw->message( preOpenBuffer, LogWidget::LOG_WARNING );
      preOpenBuffer.clear();
    }

    lw->message( message, severity );
  }
  else
  {
    preOpenBuffer += message;
  }
}

void Viewer::emitContinuousUpdate()
{
  emit continuousUpdate();
}

void Viewer::emitViewChanged()
{
  emit viewChanged();
}

void Viewer::emitMaterialChanged()
{
  emit materialChanged();
}

void Viewer::emitSceneTreeChanged()
{
  emit sceneTreeChanged();
}

// used in the script system..
void Viewer::runEventLoop()
{
  processEvents();
}

void Viewer::executeCommand( ViewerCommand * command )
{
  if( command->isParameterCommand() )
  {
    m_parameterUndoStack.push( command );
  }
  else
  {
    m_sceneStateUndoStack.push( command );
    // we always clear the parameter stack on a state assignment
    // probably not required when adding light sources...
    m_parameterUndoStack.clear();
  }
}

// outputs statistics on current scene
void Viewer::outputStatistics()
{
  if ( m_viewState && m_viewState->getScene() )
  {
    dp::sg::algorithm::StatisticsTraverser statisticsTraverser;
    statisticsTraverser.apply( m_viewState->getScene() );
    const dp::sg::algorithm::Statistics * stats = statisticsTraverser.getStatistics();

    size_t totalVertices = stats->m_statVertexAttributeSet.m_numberOfVertices;
    size_t totalPatches = stats->m_statPrimitives.m_patches;
    DP_ASSERT( totalPatches <= stats->m_statPrimitives.m_faces );

    std::stringstream ss;
    ss << "Vertices: " << (int)totalVertices << "  Faces: " << (int) stats->m_statPrimitives.m_faces - totalPatches;
    if ( totalPatches )
    {
      ss << "  Patches: " << (int) totalPatches;
    }
    m_mainWindow->getStatisticsLabel()->setText(  QApplication::translate(VIEWER_APPLICATION_NAME, ss.str().c_str(), 0) );
  }
}

const dp::sg::core::EffectDataSharedPtr & Viewer::getEffectData( const std::string & effectName )
{
  EffectDataMap::const_iterator it = m_effectDataLibrary.find( effectName );
  if ( it == m_effectDataLibrary.end() )
  {
    it = m_effectDataLibrary.insert( make_pair( effectName, dp::sg::core::EffectData::create( dp::fx::EffectLibrary::instance()->getEffectData( effectName ) ) ) ).first;
  }
  return( it->second );
}

class IsEffectData
{
  public:
    IsEffectData( const dp::sg::core::EffectDataSharedPtr & effectData )
      : m_effectData( effectData )
    {
    }

    bool operator()( const std::pair<std::string,dp::sg::core::EffectDataSharedPtr> & data )
    {
      return( m_effectData == data.second );
    }

  private:
    const dp::sg::core::EffectDataSharedPtr & m_effectData;
};

bool Viewer::holdsEffectData( const dp::sg::core::EffectDataSharedPtr & effectData )
{
  return( std::find_if( m_effectDataLibrary.cbegin(), m_effectDataLibrary.cend(), IsEffectData( effectData ) ) != m_effectDataLibrary.cend() );
}
