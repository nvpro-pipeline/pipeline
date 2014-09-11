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


QString appStyleSheet =
"\
QWidget\
{\
 color:#e1e1e1;\
 background-color:#4b4b4b;\
 alternate-background-color:#505050;\
 selection-color:#000000;\
 selection-background-color:#b4b4b4;\
}\
QWidget:disabled\
{\
 color:#808080;\
}\
QMenuBar\
{\
 color:#e1e1e1;\
 background-color:#161616;\
 selection-color:#000000;\
 selection-background-color:#b4b4b4;\
}\
QMenuBar::item\
{\
 color:#e1e1e1;\
 background-color:#161616;\
 selection-color:#000000;\
 selection-background-color:#b4b4b4;\
 border:1px solid transparent;\
 border-radius:0px;\
 padding:1px 5px 1px 5px;\
}\
/* Hover doesn't work on this. */\
QMenuBar::item:selected\
{\
 color:#000000;\
 background-color:#b4b4b4;\
 border-style:outset;\
 border-color:#969696;\
}\
QMenu\
{\
 background-color:#303030;\
 margin:1px;\
 border:1px solid #5a5a5a;\
}\
QMenu::item\
{\
 padding:2px 20px 2px 20px;\
 border:1px solid transparent;\
}\
QMenu::item:selected\
{\
 color:#000000;\
 background-color:#b4b4b4;\
 border-style:outset;\
 border-color:#969696;\
}\
QMenu::icon:checked\
{\
 background-color:#4b4b4b;\
 border-width:1px;\
 border-style:inset;\
 border-color:#3c3c3c #4b4b4b #4b4b4b #3c3c3c;\
 position:absolute;\
 top:1px;\
 right:1px;\
 bottom:1px;\
 left:1px;\
}\
QMenu::separator\
{\
 height:0px;\
 border:1px inset #303030;\
 margin-left:5px;\
 margin-right:5px;\
}\
QMenu::indicator\
{\
 width:20px;\
 height:20px;\
}\
QDockWidget\
{\
  titlebar-close-icon:url(:/images/DockClose.png);\
  titlebar-normal-icon:url(:/images/DockFloat.png);\
}\
QDockWidget::title\
{\
 text-align:left;\
 padding-left:5px;\
 padding-right:32px;\
 background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0.0 #141414, stop:1.0 #1a1a1a);\
 border-width:1px;\
 border-style:solid;\
 border-color:#212121 #121212 #161616 #161616;\
}\
QToolTip\
{\
 color:#ffffff;\
 background:#969696;\
 border-color:#5a5a5a;\
 border-radius:5px;\
}\
QGroupBox\
{\
 border:2px groove #3c3c3c;\
 border-radius:5px;\
 margin-top:0.5em;\
 padding:3px;\
}\
QGroupBox::title\
{\
 subcontrol-origin:margin;\
 margin-left:2px;\
 left:8px;\
}\
QLineEdit,\
QPlainTextEdit,\
QTextEdit,\
QSpinBox,\
QDoubleSpinBox\
{\
 background-color:#5a5a5a;\
 border-width:1px;\
 border-style:inset;\
 border-color:#3c3c3c #4b4b4b #4b4b4b #3c3c3c;\
 border-radius:5px;\
}\
QPushButton\
{\
 border-image:url(:/images/PushButton.png) 3;\
 border-width:3px;\
 padding-left:4px;\
 padding-right:4px;\
 min-width:8ex;\
}\
QPushButton:hover\
{\
 border-image:url(:/images/PushButton_hover.png) 3;\
}\
QPushButton:default\
{\
 border-image:url(:/images/PushButton_default.png) 3;\
}\
QPushButton:default:hover\
{\
 border-image:url(:/images/PushButton_default_hover.png) 3;\
}\
QPushButton:pressed,\
QPushButton:default:pressed\
{\
 border-image:url(:/images/PushButton_pressed.png) 3;\
}\
QPushButton:disabled\
{\
 border-image:url(:/images/PushButton_disabled.png) 3;\
}\
QToolBar\
{\
 background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,\
  stop:0.0 #252525, stop:0.3 #303030, stop:0.6 #404040, stop:1.0 #4b4b4b);\
 border:1px solid;\
 border-color:#5a5a5a #3c3c3c #252525 #3c3c3c;\
 padding:3px;\
}\
QToolBar::separator\
{\
 border:1px inset #3c3c3c;\
}\
QToolBar::separator:horizontal\
{\
 width:0px;\
 margin-left:5px;\
 margin-right:5px;\
}\
QToolBar::separator:vertical\
{\
 height:0px;\
 margin-top:5px;\
 margin-bottom:5px;\
}\
QToolButton\
{\
 background-color:transparent;\
 border:1px solid transparent;\
 border-radius:5px;\
}\
QToolButton:hover\
{\
 color:#FFFFFF;\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
QToolButton:pressed\
{\
 color:#b4b4b4;\
 background-color:#5a5a5a;\
 border-style:inset;\
 border-color:#3c3c3c;\
}\
QComboBox\
{\
 border:1px outset #3c3c3c;\
 border-radius:5px;\
 padding:3px;\
}\
/* Qt Bug:QComboBox:!editable color does not take effect! */\
QComboBox:hover\
{\
 color:#FFFFFF;\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
QComboBox:on\
{\
 color:#b4b4b4;\
 background-color:#5a5a5a;\
 border-style:inset;\
 border-color:#3c3c3c;\
}\
QComboBox::drop-down\
{\
 subcontrol-origin:padding;\
 subcontrol-position:right;\
 width:11px;\
 /* No way to set the border color of the drop-down when hovering => drop-down border removed.\
 border-left-width:2px;\
 border-color:#3c3c3c;\
 border-left-style:groove;\
 */\
 border-top-right-radius:5px;\
 border-bottom-right-radius:5px;\
 }\
QComboBox::drop-down:hover\
{\
 border-color:#969696;\
}\
QComboBox::drop-down:on\
{\
 border-color:#3c3c3c;\
}\
QComboBox::down-arrow\
{\
 image:url(:/images/combobox_down_arrow.png);\
}\
QCheckBox\
{\
 spacing:5px;\
}\
QCheckBox::indicator,\
QGroupBox::indicator\
{\
 width:12px;\
 height:12px;\
}\
QCheckBox::indicator:unchecked,\
QGroupBox::indicator:unchecked\
{\
 image:url(:/images/CBT_Unchecked.png);\
}\
QCheckBox::indicator:unchecked:disabled,\
QGroupBox::indicator:unchecked:disabled\
{\
 image:url(:/images/CBT_UncheckedDisabled.png);\
}\
QCheckBox::indicator:unchecked:hover,\
QGroupBox::indicator:unchecked:hover\
{\
 image:url(:/images/CBT_UncheckedHover.png);\
}\
QCheckBox::indicator:unchecked:pressed,\
QGroupBox::indicator:unchecked:pressed\
{\
 image:url(:/images/CBT_UncheckedPressed.png);\
}\
QCheckBox::indicator:checked,\
QGroupBox::indicator:checked\
{\
 image:url(:/images/CBT_Checked.png);\
}\
QCheckBox::indicator:checked:disabled,\
QGroupBox::indicator:checked:disabled\
{\
 image:url(:/images/CBT_CheckedDisabled.png);\
}\
QCheckBox::indicator:checked:hover,\
QGroupBox::indicator:checked:hover\
{\
 image:url(:/images/CBT_CheckedHover.png);\
}\
QCheckBox::indicator:checked:pressed,\
QGroupBox::indicator:checked:pressed\
{\
 image:url(:/images/CBT_CheckedPressed.png);\
}\
QCheckBox::indicator:indeterminate,\
QGroupBox::indicator:indeterminate\
{\
 image:url(:/images/checkbox_indeterminate.png);\
}\
QCheckBox::indicator:indeterminate:hover,\
QGroupBox::indicator:indeterminate:hover\
{\
 image:url(:/images/checkbox_indeterminate_hover.png);\
}\
QCheckBox::indicator:indeterminate:pressed,\
QGroupBox::indicator:indeterminate:pressed\
{\
 image:url(:/images/checkbox_indeterminate_pressed.png);\
}\
QRadioButton::indicator\
{\
 width:12px;\
 height:12px;\
}\
QRadioButton::indicator::unchecked\
{\
 image:url(:/images/RB_Unchecked.png);\
}\
QRadioButton::indicator::unchecked:disabled\
{\
 image:url(:/images/RB_UncheckedDisabled.png);\
}\
QRadioButton::indicator:unchecked:hover\
{\
 image:url(:/images/RB_UncheckedHover.png);\
}\
QRadioButton::indicator:unchecked:pressed\
{\
 image:url(:/images/RB_UncheckedPressed.png);\
}\
QRadioButton::indicator::checked\
{\
 image:url(:/images/RB_Checked.png);\
}\
QRadioButton::indicator::checked:disabled\
{\
 image:url(:/images/RB_CheckedDisabled.png);\
}\
QRadioButton::indicator:checked:hover\
{\
 image:url(:/images/RB_CheckedHover.png);\
}\
QRadioButton::indicator:checked:pressed\
{\
 image:url(:/images/RB_CheckedPressed.png);\
}\
QScrollBar:horizontal\
{\
 border:0px;\
 margin:0px 16px 0px 16px;\
 min-width:16px;\
}\
QScrollBar:vertical\
{\
 border:0px;\
 margin:16px 0px 16px 0px;\
 min-height:16px;\
}\
QScrollBar::handle:horizontal\
{\
 border:1px outset #3c3c3c;\
 min-width:14px;\
}\
QScrollBar::handle:vertical\
{\
 border:1px outset #3c3c3c;\
 min-height:14px;\
}\
QScrollBar::sub-line:horizontal\
{\
 subcontrol-origin:margin;\
 subcontrol-position:left;\
 border:1px outset #3c3c3c;\
 width:14px;\
}\
QScrollBar::sub-line:vertical\
{\
 subcontrol-origin:margin;\
 subcontrol-position:top;\
 border:1px outset #3c3c3c;\
 height:14px;\
}\
QScrollBar::add-line:horizontal\
{\
 subcontrol-origin:margin;\
 subcontrol-position:right;\
 border:1px outset #3c3c3c;\
 width:14px;\
}\
QScrollBar::add-line:vertical\
{\
 subcontrol-origin:margin;\
 subcontrol-position:bottom;\
 border:1px outset #3c3c3c;\
 height:14px;\
}\
QScrollBar::sub-page,\
QScrollBar::add-page\
{\
 background-color:#3c3c3c;\
}\
QScrollBar::sub-line:horizontall\
{\
 image:url(:/images/scroll_left_arrow.png);\
}\
QScrollBar::add-line:horizontal\
{\
 image:url(:/images/scroll_right_arrow.png);\
}\
QScrollBar::sub-line:vertical\
{\
 image:url(:/images/scroll_up_arrow.png);\
}\
QScrollBar::add-line:vertical\
{\
 image:url(:/images/scroll_down_arrow.png);\
}\
QScrollBar::sub-line:horizontal:hover\
{\
 image:url(:/images/scroll_left_arrow_hover.png);\
}\
QScrollBar::add-line:horizontal:hover\
{\
 image:url(:/images/scroll_right_arrow_hover.png);\
}\
QScrollBar::sub-line:vertical:hover\
{\
 image:url(:/images/scroll_up_arrow_hover.png);\
}\
QScrollBar::add-line:vertical:hover\
{\
 image:url(:/images/scroll_down_arrow_hover.png);\
}\
QScrollBar::sub-line:horizontal:hover:pressed\
{\
 image:url(:/images/scroll_left_arrow_pressed.png);\
}\
QScrollBar::add-line:horizontal:hover:pressed\
{\
 image:url(:/images/scroll_right_arrow_pressed.png);\
}\
QScrollBar::sub-line:vertical:hover:pressed\
{\
 image:url(:/images/scroll_up_arrow_pressed.png);\
}\
QScrollBar::add-line:vertical:hover:pressed\
{\
 image:url(:/images/scroll_down_arrow_pressed.png);\
}\
QListView,\
QListWidget,\
QTreeView,\
QTreeWidget\
{\
 border:none;\
}\
QListView::item,\
QListWidget::item,\
QTreeView::item,\
QTreeWidget::item\
{\
 border:1px outset transparent;\
 border-radius:5px;\
}\
QListView::item:hover,\
QListWidget::item:hover,\
QTreeView::item:hover,\
QTreeWidget::item:hover\
{\
 color:#000000;\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
QListView::item:selected:active,\
QListWidget::item:selected:active,\
QTreeView::item:selected:active,\
QTreeWidget::item:selected:active\
{\
 color:#e1e1e1;\
 background-color:#969696;\
 border-color:#787878;\
}\
QListView::item:selected:active:hover,\
QListWidget::item:selected:active:hover,\
QTreeView::item:selected:active:hover,\
QTreeWidget::item:selected:active:hover\
{\
 color:#FFFFFF;\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
QListView::item:selected:!active,\
QListWidget::item:selected:!active,\
QTreeView::item:selected:!active,\
QTreeWidget::item:selected:!active\
{\
 color:#e1e1e1;\
 background-color:#5a5a5a;\
 border-color:#3c3c3c;\
}\
QListView::item:selected:!active:hover,\
QListWidget::item:selected:!active:hover,\
QTreeView::item:selected:!active:hover,\
QTreeWidget::item:selected:!active:hover\
{\
 color:#e1e1e1;\
 background-color:#969696;\
 border-color:#787878;\
}\
QTreeView::branch:has-children:!has-siblings:closed,\
QTreeView::branch:closed:has-children:has-siblings\
{\
 border-image:none;\
 image:url(:/images/TV_BranchClosed.png);\
}\
QTreeView::branch:has-children:!has-siblings:closed:hover,\
QTreeView::branch:closed:has-children:has-siblings:hover\
{\
 border-image:none;\
 image:url(:/images/TV_BranchClosedHover.png);\
}\
QTreeView::branch:open:has-children:!has-siblings,\
QTreeView::branch:open:has-children:has-siblings\
{\
 border-image:none;\
 image:url(:/images/TV_BranchOpened.png);\
}\
QTreeView::branch:open:has-children:!has-siblings:hover,\
QTreeView::branch:open:has-children:has-siblings:hover\
{\
 border-image:none;\
 image:url(:/images/TV_BranchOpenedHover.png);\
}\
QSpinBox::up-button,\
QDoubleSpinBox::up-button\
{\
 subcontrol-origin:border;\
 subcontrol-position:top right;\
 border-width:0px;\
 border-top-right-radius:5px;\
}\
QSpinBox::up-arrow,\
QDoubleSpinBox::up-arrow\
{\
 image:url(:/images/spin_up_arrow.png);\
}\
QSpinBox::up-arrow:hover,\
QDoubleSpinBox::up-arrow:hover\
{\
 image:url(:/images/spin_up_arrow_hover.png);\
}\
QSpinBox::up-arrow:pressed,\
QDoubleSpinBox::up-arrow:pressed\
{\
 image:url(:/images/spin_up_arrow_pressed.png);\
}\
QSpinBox::up-arrow:disabled,\
QSpinBox::up-arrow:off,\
QDoubleSpinBox::up-arrow:disabled,\
QDoubleSpinBox::up-arrow:off\
{\
 image:url(:/images/spin_up_arrow_off.png);\
}\
QSpinBox::down-button,\
QDoubleSpinBox::down-button\
{\
 subcontrol-origin:border;\
 subcontrol-position:bottom right;\
 border-width:0px;\
 border-bottom-right-radius:5px;\
}\
QSpinBox::down-arrow,\
QDoubleSpinBox::down-arrow\
{\
 image:url(:/images/spin_down_arrow.png);\
}\
QSpinBox::down-arrow:hover,\
QDoubleSpinBox::down-arrow:hover\
{\
 image:url(:/images/spin_down_arrow_hover.png);\
}\
QSpinBox::down-arrow:pressed,\
QDoubleSpinBox::down-arrow:pressed\
{\
 image:url(:/images/spin_down_arrow_pressed.png);\
}\
QSpinBox::down-arrow:disabled,\
QSpinBox::down-arrow:off,\
QDoubleSpinBox::down-arrow:disabled,\
QDoubleSpinBox::down-arrow:off\
{\
 image:url(:/images/spin_down_arrow_off.png);\
}\
QToolBox::tab\
{\
 color:#b4b4b4;\
 border-width:1px;\
 border-style:outset;\
 border-color:#3c3c3c;\
 border-top-left-radius:5px;\
 border-top-right-radius:5px;\
 padding:3px;\
 min-width:8ex;\
}\
QToolBox::tab:hover\
{\
 color:#000000;\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
QToolBox::tab:selected\
{\
 color:#e1e1e1;\
 background-color:#969696;\
 border-color:#787878;\
}\
QToolBox::tab:selected:hover\
{\
 color:#FFFFFF;\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
QSlider::groove:horizontal\
{\
 background-color:#3c3c3c;\
 border-width:1px;\
 border-style:inset;\
 border-color:#2d2d2d #3c3c3c #3c3c3c #2d2d2d;\
 height:2px;\
}\
QSlider::groove:horizontal:disabled\
{\
 background-color:#b4b4b4;\
 border-width:0px;\
}\
QSlider::handle:horizontal\
{\
 image:url(:/images/SliderHandle_Horizontal.png);\
 width:16px;\
 margin:-7px -5px;\
}\
QSlider::handle:horizontal:disabled\
{\
 image:url(:/images/SliderHandle_HorizontalDisabled.png);\
}\
QSlider::handle:horizontal:hover\
{\
 image:url(:/images/SliderHandle_HorizontalHover.png);\
}\
QSlider::groove:vertical\
{\
 background-color:#3c3c3c;\
 border-width:1px;\
 border-style:inset;\
 border-color:#2d2d2d #3c3c3c #3c3c3c #2d2d2d;\
 width:2px;\
}\
QSlider::groove:vertical\
{\
 background-color:#b4b4b4;\
 border-width:0px;\
}\
QSlider::handle:vertical\
{\
 image:url(:/images/SliderHandle_Vertical.png);\
 width:16px;\
 margin:-5px -7px;\
}\
QSlider::handle:vertical:disabled\
{\
 image:url(:/images/SliderHandle_VerticalDisabled.png);\
}\
QSlider::handle:vertical:hover\
{\
 image:url(:/images/SliderHandle_VerticalHover.png);\
}\
QTableCornerButton::section,\
QHeaderView::section\
{\
 color:#e1e1e1;\
 background-color:#3c3c3c;\
 border-width:1px;\
 border-style:outset;\
 border-color:#3c3c3c #2d2d2d #2d2d2d #3c3c3c;\
 border-radius:5px;\
 padding:0px 3px;\
}\
QHeaderView::section:checked\
{\
 color:#b4b4b4;\
 background-color:#5a5a5a;\
 border-style:inset;\
 border-color:#3c3c3c;\
 font-weight:normal;\
}\
QHeaderView::down-arrow\
{\
 image:url(:/images/spin_down_arrow.png);\
}\
QHeaderView::up-arrow\
{\
 image:url(:/images/spin_up_arrow.png);\
}\
QStatusBar\
{\
 background-color:#161616;\
 border:1px outset #4b4b4b;\
}\
QTabWidget::pane\
{\
 border-width:1px;\
 border-style:outset;\
 border-color:#3c3c3c;\
}\
QTabWidget::tab-bar\
{\
 left:5px;\
 top:1px;\
}\
QTabWidget::tab-bar:bottom\
{\
 top:-1px;\
}\
QTabBar::tab:top\
{\
 color:#969696;\
 border-image:url(:/images/tab_top_border_unselected.png) 4 7 4 7;\
 border-width:4px 7px 4px 7px;\
 padding-left:4px;\
 padding-right:4px;\
 min-width:8ex;\
}\
QTabBar::tab:top:hover\
{\
 color:#000000;\
 border-image:url(:/images/tab_top_border_unselected_hover.png) 4 7 4 7;\
}\
QTabBar::tab:top:selected\
{\
 color:#e1e1e1;\
 border-image:url(:/images/tab_top_border_selected.png) 4 7 4 7;\
}\
QTabBar::tab:bottom\
{\
 color:#969696;\
 border-image:url(:/images/tab_bottom_border_unselected.png) 4 7 4 7;\
 border-width:4px 7px 4px 7px;\
 padding-left:4px;\
 padding-right:4px;\
 min-width:8ex;\
}\
QTabBar::tab:bottom:hover\
{\
 color:#000000;\
 border-image:url(:/images/tab_bottom_border_unselected_hover.png) 4 7 4 7;\
}\
QTabBar::tab:bottom:selected\
{\
 color:#e1e1e1;\
 border-image:url(:/images/tab_bottom_border_selected.png) 4 7 4 7;\
}\
QTabBar::tab:top:selected:hover,\
QTabBar::tab:bottom:selected:hover\
{\
 color:#FFFFFF;\
 /* No border image. Breaks the look.*/\
}\
QMainWindow::separator,\
QSplitter::handle\
{\
 border:1px outset #3c3c3c;\
}\
QMainWindow::separator:hover,\
QSplitter::handle:pressed /* QSplitter::handle:hover doesn't take effect. */\
{\
 background-color:#b4b4b4;\
 border-color:#969696;\
}\
";

