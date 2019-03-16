/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package detecçãoFaces;

import java.awt.event.KeyEvent;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

/**
 *
 * @author luis
 */
public class DetectaFaces
{
    public static void main(String args[]) throws FrameGrabber.Exception
    {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);//numero da camera instalada
        camera.start();
        
        
        CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\haarcascade-frontalface-alt.xml");
        
        
        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma()/camera.getGamma());//divisao recomendada na documentação
        
        Frame frameCapturado = null;
        Mat imagemcColorida = new Mat();
        
        while((frameCapturado = camera.grab()) != null)
        {
            imagemcColorida = convertMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();//otimozar depois
            
            cvtColor(imagemcColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();//otimozar depois
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));//analisa a imagem em sinza e armazena em facesdetectadas, bem, as faces detectadas, os ultimos 2 são tamanho minimo e tamanho maximo
            
            
            for (int i = 0; i < facesDetectadas.size(); i++)
            {
                Rect dadosFace = facesDetectadas.get(0);
                rectangle(imagemcColorida, dadosFace, new Scalar(0, 0, 255, 0));
            }
            if(cFrame.isVisible())
                cFrame.showImage(frameCapturado);
        }
        
        cFrame.dispose();//limpa memória
        camera.stop();//para  a captura
    }
}
