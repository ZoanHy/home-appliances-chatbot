import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# Create the Cypher QA chain

cypher_generation_template = """
Nhiệm vụ:
Tạo truy vấn Cypher cho cơ sở dữ liệu đồ thị Neo4j. Đảm bảo tạo câu truy vấn đúng với cấu trúc trong neo4j đầy đủ theo thứ tự và khả năng suy luận.

Hướng dẫn:
Chỉ sử dụng các loại mối quan hệ và thuộc tính đã được cung cấp trong sơ đồ.
Không sử dụng bất kỳ loại mối quan hệ hoặc thuộc tính nào không có trong sơ đồ.
Hiện về website chỉ có ba trang: dienmayxanh, nguyenkim và meta đảm bảo đưa về cho đúng

Sơ đồ:
{schema}

Lưu ý:
Không bao gồm bất kỳ lời giải thích hay lời xin lỗi nào trong phản hồi của bạn.
Không trả lời bất kỳ câu hỏi nào có thể yêu cầu bạn làm gì khác ngoài việc xây dựng một câu lệnh Cypher. 
Không bao gồm bất kỳ văn bản nào khác ngoài câu lệnh Cypher đã tạo. 
Đảm bảo rằng hướng của mối quan hệ là chính xác trong các truy vấn của bạn. 
Đảm bảo bạn khai báo bí danh cho cả hai thực thể và mối quan hệ một cách chính xác. 
Không chạy bất kỳ truy vấn nào có thể thêm vào hoặc xóa khỏi cơ sở dữ liệu.
Đảm bảo đặt bí danh cho tất cả các câu lệnh tiếp theo như với tuyên bố (Ví dụ: WITH p as Product, c.categoryID as category_id)
Nếu bạn cần chia số, hãy đảm bảo lọc mẫu số để không bằng 0.
Nếu người dùng yêu cầu thông tin chi tiết hoặc khi hỏi về giá hoặc đánh giá cao nhất hay thấp nhất bắt buộc phải trả thêm giá trị product_specifications

Ví dụ:
# Lò nướng có giá dưới 1 triệu xuất xứ tại Việt Nam và có đánh giá tốt
MATCH (p:Product)-[:BELONGS_TO]->(c:Category),
      (p)-[:BELONGS_TO]->(b:Brand)
WHERE c.name = 'Lò nướng' 
  AND toFloat(p.product_price) < 1000000
  AND p.product_specifications CONTAINS 'Việt Nam'
  AND p.product_rating > 4
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating

# Liệt kê các sản phẩm là tủ lạnh thuộc thương hiệu Toshiba.
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand), 
      (p)-[:BELONGS_TO]->(c:Category)
WHERE b.name = 'Toshiba' AND (p.product_name CONTAINS 'Tủ lạnh' OR c.name = 'Tủ lạnh')
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.website AS website,
       p.product_img_link AS product_img_link, 
       p.product_rating AS product_rating

# Liệt kê 5 sản phẩm tủ lạnh có mức giá cao nhất.
MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
WHERE c.name = 'Tủ lạnh' OR p.product_name CONTAINS 'Tủ lạnh'
RETURN p.product_name AS product_name, p.product_price AS product_price, p.website AS website
ORDER BY product_price DESC
LIMIT 5

# Liệt kê những sản phẩm tủ lạnh thuộc thương hiệu Toshiba và Samsung.
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand), (p)-[:BELONGS_TO]->(c:Category)
WHERE c.name = 'Tủ lạnh' AND (b.name = 'Toshiba' OR b.name = 'Samsung')
RETURN p.product_name, p.product_id, p.product_price, p.product_link, p.product_img_link, p.website AS website

# Có tổng cộng bao nhiêu hãng tủ lạnh và liệt kê.
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand), (p)-[:BELONGS_TO]->(c:Category)
WHERE c.name = 'Tủ lạnh' OR p.product_name CONTAINS 'Tủ lạnh'
RETURN DISTINCT b.name AS brand

# Liệt kê sản phẩm tủ lạnh hãng Toshiba giá trên 5 triệu và dưới 10 triệu
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand), 
      (p)-[:BELONGS_TO]->(c:Category)
WHERE b.name = 'Toshiba' AND c.name = 'Tủ lạnh' AND toFloat(p.product_price) > 5000000 AND toFloat(p.product_price) < 10000000
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating

# Liệt kê sản phẩm tủ lạnh hãng Toshiba giá trên 5 triệu và dưới 10 triệu
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand), 
      (p)-[:BELONGS_TO]->(c:Category)
WHERE b.name = 'Toshiba' AND c.name = 'Tủ lạnh' AND toFloat(p.product_price) > 5000000 AND toFloat(p.product_price) < 10000000 AND p.product_specifications CONTAINS '180 lít'
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating

# Liệt kê các sản phẩm máy lọc nước có 11 lõi lọc
MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
WHERE c.name = 'Máy lọc nước' AND p.product_specifications CONTAINS '11 lõi lọc'
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating

# Liệt kê các sản phẩm máy lọc nước gồm 11 lõi lọc, có công nghệ kháng khuẩn Nano Silver và sản xuất tại Việt Nam
MATCH (p:Product)-[:BELONGS_TO]->(c:Category),
      (p)-[:BELONGS_TO]->(b:Brand)
WHERE c.name = 'Máy lọc nước' 
  AND p.product_specifications CONTAINS '11 lõi lọc'
  AND p.product_specifications CONTAINS 'Nano Silver'
  AND p.product_specifications CONTAINS 'Việt Nam'
  AND b.name = 'Sunhouse'
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating

# Giá tủ lạnh cao nhất là bao nhiêu?
MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
WHERE toLower(c.name) = toLower('Tủ lạnh')
WITH p, toFloat(p.product_price) AS price
ORDER BY price DESC
LIMIT 1
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating,
       p.product_specifications as product_specifications

# Liệt kê top 10 sản phẩm máy lọc nước giá cao nhất thuộc trang website dienmayxanh
MATCH (p:Product)-[:LISTED_ON]->(w:Website),
      (p)-[:BELONGS_TO]->(c:Category),
      (p)-[:BELONGS_TO]->(b:Brand)
WHERE w.name = 'dienmayxanh' AND c.name = 'Máy lọc nước'
WITH p, toFloat(p.product_price) AS price
ORDER BY price DESC
LIMIT 10
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link,
       p.website AS website,
       p.product_rating AS product_rating
# Tài chỉnh có 5 triệu gợi ý cho tôi một vài sản phẩm tủ lạnh.
MATCH (p:Product)-[:BELONGS_TO]->(c:Category),
      (p)-[:BELONGS_TO]->(b:Brand)
WHERE c.name = 'Tủ lạnh' AND toFloat(p.product_price) <= 5000000 
RETURN p.product_name AS product_name, 
       p.product_price AS product_price, 
       p.product_link AS product_link, 
       p.product_img_link AS product_img_link, 
       p.website AS website,
       p.product_rating AS product_rating
# Tổng cộng bao nhiêu hãng máy lọc nước ở website dienmayxanh.
MATCH (p:Product)-[:BELONGS_TO]->(b:Brand), 
      (p)-[:BELONGS_TO]->(c:Category),
      (p)-[:LISTED_ON]->(w:Website)
WHERE w.name = 'dienmayxanh' AND c.name = 'Máy lọc nước'
RETURN DISTINCT b.name AS brand

Giá trị danh mục chuỗi:
Sử dụng các chuỗi và giá trị hiện có từ lược đồ được cung cấp.
Lưu ý: Phải ghi nhận hết toàn bộ kết quả từ truy vấn để đảm bảo sự đầy đủ khớp với cơ sở dữ liệu.
Sau khi tạo xong hãy tự sửa lỗi lại theo đúng thứ tự của câu truy vấn như MATCH rồi mới đến WHERE rồi đến RETURN không được xếp lộn xộn. Đảm bảo đúng cấu trúc của câu lệnh cypher.

Câu trả lời là:
{question}
"""

qa_generation_template_str = """
Bạn là một trợ lý, người sẽ tiếp nhận kết quả từ câu truy vấn Cypher của Neo4j và chuyển nó thành câu trả lời dễ hiểu. Phần kết quả truy vấn chứa thông tin từ câu truy vấn Cypher mà bạn đã tạo ra dựa trên câu hỏi tự nhiên của người dùng. Thông tin được cung cấp là chính xác và đáng tin cậy; bạn không bao giờ được nghi ngờ hay thay đổi nó bằng kiến thức của bản thân. Hãy làm cho câu trả lời trở nên dễ hiểu và dễ tiếp cận với người đọc.

Những kết quả truy vấn:
{context}
Câu hỏi:
{question}

Lưu ý: phải sử dụng toàn bộ kết quả ngữ cảnh truy vấn đưa vào vì muốn đảm bảo kết quả trả về cho người dùng phải đẩy đủ, đúng và chính xác.

Nếu phần thông tin được cung cấp là trống (được đánh dấu là []), bạn sẽ trả lời rằng bạn không biết câu trả lời.
Nếu thông tin không trống, bạn phải cung cấp câu trả lời dựa trên kết quả có sẵn. Nếu câu hỏi liên quan đến thời gian, giả sử rằng kết quả truy vấn đang sử dụng đơn vị là ngày, trừ khi có yêu cầu khác.
Khi tên các thực thể được cung cấp trong kết quả truy vấn (chẳng hạn như tên bệnh viện), hãy cẩn thận với các tên chứa dấu phẩy hoặc ký tự đặc biệt. Ví dụ, "Jones, Brown and Murray" là tên của một bệnh viện duy nhất, không phải nhiều bệnh viện. Hãy đảm bảo rằng các tên trong danh sách được trình bày rõ ràng và dễ hiểu, để không gây nhầm lẫn. Đừng bao giờ nói rằng bạn thiếu thông tin nếu có dữ liệu có sẵn từ kết quả truy vấn. Luôn sử dụng dữ liệu đã cung cấp.
Đảm bảo hãy sử dụng đầy đủ kết quả đã truy vấn và đừng giới hạn.
Nếu những câu hỏi dạng so sánh hoặc tham khảo chi tiết hãy phân tích dựa trên đặc tính kĩ thuật và giải thích được tại sao lại nên chọn hãng đó hoặc sản phẩm đó.
Lưu ý khi hiển thị chi tiết sản phẩm hay gom cụm sản phẩm theo từng website trước để đề xuất cho người dùng và nêu rõ sản phẩm thuộc website nào.

Chú ý: 
- Trả lời bằng tiếng Việt vì người dùng là người Việt Nam.
- Nếu trả lời về sản phẩm thì luôn đi kèm thông tin hình ảnh và đường dẫn sản phẩm cho ngườ dùng.
- Đảm bảo câu trả lời một cách tự nhiên, tạo thiện cảm cho người dùng và có thể sử dụng thêm cproduct_name
product_brand
product_price
product_link
product_img_link
product_specifications
product_category
product_rating
websiteác emoji để tăng sự hứng thú với người dùng.
- Giá sản phẩm theo dữ liệu là giá tính theo VND.
- Sử dụng chủ ngữ là Tôi. Ví dụ: Tôi có thể giúp bạn ...
- Khi trả về đường dẫn url phải đảm bảo đường dẫn đó đúng và có thể truy cập được và hãy tự căn chỉnh sao cho hợp lý.

Câu trả lời hữu ích:
"""

cypher_generation_prompt = PromptTemplate(
    template=cypher_generation_template
)

qa_generation_prompt = PromptTemplate(
    template=qa_generation_template_str
)

cypher_qa = GraphCypherQAChain.from_llm(
    top_k=50,
    graph=graph,
    verbose=True,
    validate_cypher=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    qa_llm=llm,
    cypher_llm=llm,
    allow_dangerous_requests=True,
)
