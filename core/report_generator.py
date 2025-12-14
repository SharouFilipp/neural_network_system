# 📁 core/report_generator.py
"""
Генератор отчетов в различных форматах
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np


class ReportGenerator:    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_pdf_report(self, report_data: Dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_report_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        styles = getSampleStyleSheet()
        elements = []
        

        title = Paragraph(f"Model Training Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        model_info = report_data.get("model_info", {})
        info_data = [
            ["Model Name:", model_info.get("name", "N/A")],
            ["Model Type:", model_info.get("type", "N/A")],
            ["Created:", model_info.get("created", "N/A")],
            ["Trained:", model_info.get("trained", "N/A")]
        ]
        
        info_table = Table(info_data, colWidths=[100, 300])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (1, 0), (1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 12))
        
        results = report_data.get("results", {})
        results_title = Paragraph("Training Results", styles['Heading2'])
        elements.append(results_title)
        elements.append(Spacer(1, 6))
        
        results_data = [
            ["Accuracy:", f"{results.get('accuracy', 0):.4f}"],
            ["Loss:", f"{results.get('loss', 0):.4f}" if results.get('loss') else "N/A"]
        ]
        
        results_table = Table(results_data, colWidths=[100, 100])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 10),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(results_table)
        
        doc.build(elements)
        
        return str(filepath)
    
    def generate_csv_report(self, report_data: Dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_report_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        csv_data = []
        
        model_info = report_data.get("model_info", {})
        csv_data.append(["Section", "Field", "Value"])
        csv_data.append(["Model Info", "Name", model_info.get("name")])
        csv_data.append(["Model Info", "Type", model_info.get("type")])
        csv_data.append(["Model Info", "Created", model_info.get("created")])
        csv_data.append(["Model Info", "Trained", model_info.get("trained")])
        
        results = report_data.get("results", {})
        csv_data.append(["Results", "Accuracy", results.get("accuracy")])
        csv_data.append(["Results", "Loss", results.get("loss")])
        
        training_info = report_data.get("training_info", {})
        hyperparams = training_info.get("hyperparameters", {})
        for key, value in hyperparams.items():
            csv_data.append(["Hyperparameters", key, value])
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        return str(filepath)
    
    def generate_comparison_report(self, comparison_data: Dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def generate_training_plots(self, training_history: Dict[str, List[float]]) -> List[str]:
        plot_paths = []
        
        if 'accuracy' in training_history:
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['accuracy'], label='Accuracy')
            if 'val_accuracy' in training_history:
                plt.plot(training_history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            accuracy_plot = self.output_dir / f"accuracy_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(accuracy_plot)
            plot_paths.append(str(accuracy_plot))
            plt.close()

        if 'loss' in training_history:
            plt.figure(figsize=(10, 6))
            plt.plot(training_history['loss'], label='Loss')
            if 'val_loss' in training_history:
                plt.plot(training_history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            loss_plot = self.output_dir / f"loss_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(loss_plot)
            plot_paths.append(str(loss_plot))
            plt.close()
        
        return plot_paths