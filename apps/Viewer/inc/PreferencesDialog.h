// Copyright NVIDIA Corporation 2009-2011
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

#include <QDialog>
#include <QLabel>
#include <QListWidget>
#include <QStringList>

class PreferencesDialog : public QDialog
{
  Q_OBJECT

  public:
    PreferencesDialog(QWidget * parent = nullptr);
    ~PreferencesDialog();

  public slots:
    void restore();

  protected slots:
    void addPath(bool checked);
    void removePath(bool checked);
    void moveUpPath(bool checked);
    void moveDownPath(bool checked);

    void selectEnvironmentMap( bool checked );
    void selectMaterialCatalogPath( bool checked );

  private:
    QListWidget * m_searchPaths;
    QListWidget * m_effectFiles;
    QLabel      * m_environmentMapLabel;
    QLabel      * m_materialCatalogLabel;

    QStringList m_restoreSearchPaths;
    QString     m_restoreEnvironmentTextureName;
    QString     m_restoreMaterialCatalogPath;
};
