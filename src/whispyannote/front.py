# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'front.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QLabel,
    QMainWindow, QMenuBar, QPushButton, QSizePolicy,
    QSpacerItem, QStatusBar, QTextBrowser, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(428, 503)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.loopSelect = QComboBox(self.centralwidget)
        self.loopSelect.setObjectName(u"loopSelect")

        self.gridLayout.addWidget(self.loopSelect, 1, 1, 1, 3)

        self.toggle = QPushButton(self.centralwidget)
        self.toggle.setObjectName(u"toggle")

        self.gridLayout.addWidget(self.toggle, 4, 3, 1, 1)

        self.micSelect = QComboBox(self.centralwidget)
        self.micSelect.setObjectName(u"micSelect")

        self.gridLayout.addWidget(self.micSelect, 0, 1, 1, 3)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName(u"textBrowser")

        self.gridLayout.addWidget(self.textBrowser, 2, 0, 1, 4)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 4, 0, 1, 3)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 3, 0, 1, 4)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 428, 33))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"whispyannote", None))
        self.toggle.setText(QCoreApplication.translate("MainWindow", u"\u25b7", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"mic:", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"system audio:", None))
        self.label.setText("")
    # retranslateUi

