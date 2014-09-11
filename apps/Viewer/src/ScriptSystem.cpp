
// Copyright NVIDIA Corporation 2009-2010
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


#include <QFile>
#include <QTextStream>
#include "ScriptSystem.h"
#include "Log.h"

ScriptSystem::ScriptSystem(QObject *parent)
 : QScriptEngine( parent )
{
}

ScriptSystem::~ScriptSystem()
{
}

QScriptValue 
ScriptSystem::addObject( const QString & name, QObject * object )
{
  QScriptValue value = newQObject( object );
  globalObject().setProperty( name, value );

  return value;
}

QScriptValue 
ScriptSystem::addSubObject( QScriptValue & parent, const QString & name, QObject * object )
{
  QScriptValue value = newQObject( object );
  parent.setProperty( name, value );

  return value;
}

void 
ScriptSystem::addFunction( const QString & name, const QScriptEngine::FunctionSignature & func )
{
  QScriptValue fun = newFunction( func );
  globalObject().setProperty( name, fun );
}

QScriptValue 
ScriptSystem::executeScript( const QString & program, const QString & context ) 
{
  QScriptValue result = evaluate( program, context );
  if( hasUncaughtException() )
  {
    int line =uncaughtExceptionLineNumber();
    LogError( "Script Exception: (%d): %s\n", line, result.toString().toStdString().c_str() );

    return QScriptValue( QScriptValue::UndefinedValue );
  }
  else
  {
    return result;
  }
}

QScriptValue 
ScriptSystem::executeFile( const QString & path, const QString & context ) 
{
  QFile file( path );
  if( !file.open(QIODevice::ReadOnly) )
  {
    LogError( "Error opening script: \"%s\"\n", path.toStdString().c_str() );
    return QScriptValue( QScriptValue::UndefinedValue );
  }

  QTextStream stream(&file);
  QString contents = stream.readAll();
  file.close();

  return executeScript( contents, context );
}

