/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reconhecimento;

import java.awt.event.KeyEvent;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
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
public class Reconhecimento
{
    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException
    {
        KeyEvent tecla = null;
        boolean flag = true;
        OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);//numero da camera instalada
        String []pessoas = {"", "Luis", "Gustavo"};//primeira posicao vazia porque lá em baixo ele retorna 1 ou 2
        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\haarcascade-frontalface-alt.xml");
        
        
        
        
        /*
        //EigenFace
        FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
        reconhecedor.read("src\\recursos\\classificadorEigenFaces.yml");//carrega o arquivo treinado
        reconhecedor.setThreshold(5000);//se o valor da confiabilidade for maior que o parametro passado automaticamente tornace desconhecido - linha opcional
        */
        
        /*
        //FisherFace
        FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
        reconhecedor.read("src\\recursos\\classificadorFisherFaces.yml");//carrega o arquivo treinado
        */
        
        //LBPH
        FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
        reconhecedor.read("src\\recursos\\classificadorLBPHFaces.yml");//carrega o arquivo treinado
        
        
        
        
        
        
        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());//divisao recomendada na documentação

        Frame frameCapturado = null;
        Mat imagemcColorida = new Mat();

        /*System.out.println("Digite Seu ID: ");
        Scanner cadastro = new Scanner(System.in);
        int idpessoa = cadastro.nextInt();*/
        while ((frameCapturado = camera.grab()) != null && flag)
        {
            imagemcColorida = convertMat.convert(frameCapturado);
            Mat imagemCinza = new Mat();//otimozar depois

            cvtColor(imagemcColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();//otimozar depois
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));//analisa a imagem em sinza e armazena em facesdetectadas, bem, as faces detectadas, os ultimos 2 são tamanho minimo e tamanho maximo

            if (tecla == null)
                tecla = cFrame.waitKey(5);//espera pra ver se o usuario pressionou algo
            if (tecla != null)
                {
                    if (tecla.getKeyChar() == 27)
                    {
                        flag = false;
                    }
                }
            for (int i = 0;flag && i < facesDetectadas.size(); i++)
            {
                Rect dadosFace = facesDetectadas.get(i);
                rectangle(imagemcColorida, dadosFace, new Scalar(0, 255, 0, 0));

                Mat faceCapturada = new Mat(imagemCinza, dadosFace);//jogando pra dentro de faceCapturada somente a parte da face da imagem
                resize(faceCapturada, faceCapturada, new Size(160, 160));//é recomedado na documentação que todas as imagens tenham o mesmo tamanho, afim de facilitar a eficiencia na comparação
                
                
                IntPointer rotulo = new IntPointer(1);//rotulo da imagem
                DoublePointer confianca = new DoublePointer(1);//confiabilidade do reconhecimento
                reconhecedor.predict(faceCapturada, rotulo, confianca);
                int predicao = rotulo.get(0);//resposta final ID de quem foi encontrado
                
                String nome;
                if(predicao == -1)//não achou nenhuma classe
                {
                    nome = "desconhecido";
                }
                else
                {
                    nome = pessoas[predicao] + " - " + confianca.get(0);
                }
                
                int x = Math.max(dadosFace.tl().x()-10, 0);
                int y = Math.max(dadosFace.tl().y()-10, 0);
                putText(imagemcColorida, nome, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0, 255, 0, 0));
            }
            if (cFrame.isVisible())
            {
                cFrame.showImage(frameCapturado);
            }
        }

        cFrame.dispose();//limpa memória
        camera.stop();//para  a captura
    }
}
