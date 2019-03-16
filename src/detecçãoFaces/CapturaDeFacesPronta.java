/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package detecçãoFaces;


import java.awt.event.KeyEvent;
import java.util.Scanner;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
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
public class CapturaDeFacesPronta
{
    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException
    {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);//numero da camera instalada
        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\haarcascade-frontalface-alt.xml");

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());//divisao recomendada na documentação

        Frame frameCapturado = null;
        Mat imagemcColorida = new Mat();
        int numeroAmostras = 25;//25 fotos numero recomendado pela documentação como o minimo para um bom desempenho
        int amostra = 1;//contador

        System.out.println("Digite Seu ID: ");
        Scanner cadastro = new Scanner(System.in);
        int idpessoa = cadastro.nextInt();
        while ((frameCapturado = camera.grab()) != null)
        {
            imagemcColorida = convertMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();//otimozar depois

            cvtColor(imagemcColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();//otimozar depois
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));//analisa a imagem em sinza e armazena em facesdetectadas, bem, as faces detectadas, os ultimos 2 são tamanho minimo e tamanho maximo

            if (tecla == null)
                tecla = cFrame.waitKey(5);//espera pra ver se o usuario pressionou algo
            for (int i = 0; i < facesDetectadas.size(); i++)
            {
                Rect dadosFace = facesDetectadas.get(0);
                rectangle(imagemcColorida, dadosFace, new Scalar(0, 0, 255, 0));

                Mat faceCapturada = new Mat(imagemCinza, dadosFace);//jogando pra dentro de faceCapturada somente a parte da face da imagem
                resize(faceCapturada, faceCapturada, new Size(160, 160));//é recomedado na documentação que todas as imagens tenham o mesmo tamanho, afim de facilitar a eficiencia na comparação
                if (tecla == null)
                    tecla = cFrame.waitKey(5);
                if (tecla != null)
                {
                    if (tecla.getKeyChar() == 'q')
                    {
                        if (amostra <= numeroAmostras)
                        {
                            imwrite("src\\fotos\\pessoa." + idpessoa + "." + amostra + ".jpg", faceCapturada);
                            System.out.println("Foto " + amostra + " capturada\n");
                            amostra++;
                        }
                    }
                    tecla = null;
                }
            }
            if(tecla == null)
                tecla = cFrame.waitKey(20);
            if (cFrame.isVisible())
            {
                cFrame.showImage(frameCapturado);
            }
            if (amostra > numeroAmostras)
            {
                break;//trocar depois URGENTE
            }
        }

        cFrame.dispose();//limpa memória
        camera.stop();//para  a captura
    }
}
